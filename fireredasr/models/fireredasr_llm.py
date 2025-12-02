import logging
import os
import random
import re

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from fireredasr.models.fireredasr_aed import FireRedAsrAed
from fireredasr.models.module.adapter import Adapter
from fireredasr.tokenizer.llm_tokenizer import DEFAULT_SPEECH_TOKEN, IGNORE_TOKEN_ID
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper
from fireredasr.utils.param import count_model_parameters


class FireRedAsrLlm(nn.Module):
    @classmethod
    def load_encoder(cls, model_path):
        assert os.path.exists(model_path)
        package = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = FireRedAsrAed.from_args(package["args"])
        if "model_state_dict" in package:
            model.load_state_dict(package["model_state_dict"], strict=False)
        encoder = model.encoder
        encoder_dim = encoder.odim
        return encoder, encoder_dim

    @classmethod
    def from_args(cls, args):
        args.freeze_encoder = True
        args.use_flash_attn = False
        args.use_fp16 = False
        args.freeze_llm = True
        # print(f"### args={args}")
        logging.info(args)
        logging.info("Build FireRedAsrLlm")
        # Build Speech Encoder
        encoder, encoder_dim = cls.load_encoder(args.encoder_path)
        count_model_parameters(encoder)
        if args.freeze_encoder:
            logging.info(f"Frezee encoder")
            for name, param in encoder.named_parameters():
                param.requires_grad = False
            encoder.eval()

        if args.use_flash_attn:
            attn_implementation = "flash_attention_2"
            if args.use_fp16:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        else:
            attn_implementation = "eager"
            if args.use_fp16:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        
        # Build LLM
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        count_model_parameters(llm)

        # LLM Freeze or LoRA
        llm_dim = llm.config.hidden_size
        if args.freeze_llm:
            logging.info(f"Frezee LLM")
            for name, param in llm.named_parameters():
                param.requires_grad = False
            llm.eval()
        else:
            if args.use_lora:
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=64,
                    lora_alpha=16,
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "up_proj",
                        "gate_proj",
                        "down_proj",
                    ],
                    lora_dropout=0.05,
                    task_type="CAUSAL_LM",
                )
                llm = get_peft_model(llm, lora_config)
                llm.print_trainable_parameters()

        tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(args.llm_dir)
        assert tokenizer.pad_token_id == tokenizer.convert_tokens_to_ids("<|endoftext|>")
        llm.config.pad_token_id = tokenizer.pad_token_id
        llm.config.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        llm.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        llm.config.default_speech_token_id = tokenizer.convert_tokens_to_ids(
            DEFAULT_SPEECH_TOKEN
        )

        # Build projector
        encoder_projector = Adapter(
            encoder_dim, llm_dim, args.encoder_downsample_rate)
        count_model_parameters(encoder_projector)

        return cls(encoder, llm, encoder_projector,
                   args.freeze_encoder, args.freeze_llm)

    def __init__(self, encoder, llm, encoder_projector,
                 freeze_encoder, freeze_llm):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        # args
        self.freeze_encoder = freeze_encoder
        self.freeze_llm = freeze_llm
        self.llm_config = llm.config

    def transcribe(self, padded_feat, feat_lengths, padded_input_ids, attention_mask,
                   beam_size=1, decode_max_len=0, decode_min_len=0,
                   repetition_penalty=1.0, llm_length_penalty=1.0, temperature=1.0):
        encoder_outs, enc_lengths, enc_mask = self.encoder(padded_feat, feat_lengths)
        speech_features, speech_lens = self.encoder_projector(encoder_outs, enc_lengths)
        inputs_embeds = self.llm.get_input_embeddings()(padded_input_ids)

        inputs_embeds, attention_mask, _ = \
            self._merge_input_ids_with_speech_features(
                speech_features.to(inputs_embeds.dtype), inputs_embeds, padded_input_ids, attention_mask,
                speech_lens=speech_lens
            )

        max_new_tokens = speech_features.size(1) if decode_max_len < 1 else decode_max_len
        max_new_tokens = max(1, max_new_tokens)

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            do_sample=False,
            min_length=decode_min_len,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            length_penalty=llm_length_penalty,
            temperature=temperature,
            bos_token_id=self.llm.config.bos_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            pad_token_id=self.llm.config.pad_token_id,
        )

        return generated_ids


    def _merge_input_ids_with_speech_features(
        self, speech_features, inputs_embeds, input_ids, attention_mask, labels=None,
        speech_lens=None
    ):
        """
        Modified from: https://github.com/k2-fsa/icefall/blob/master/egs/speech_llm/ASR_LLM/whisper_llm_zh/model.py
        """
        speech_lens = None
        num_speechs, speech_len, embed_dim = speech_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(self.llm.config.pad_token_id)
        )
        # 1. Create a mask to know where special speech tokens are
        special_speech_token_mask = input_ids == self.llm.config.default_speech_token_id
        num_special_speech_tokens = torch.sum(special_speech_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (
            num_special_speech_tokens.max() * (speech_len - 1)
        ) + sequence_length
        batch_indices, non_speech_indices = torch.where(
            input_ids != self.llm.config.default_speech_token_id
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged speech-text sequence.
        # `special_speech_token_mask` identifies speech tokens. Each speech token will be replaced by `nb_text_tokens_per_speechs - 1` text tokens.
        # `torch.cumsum` computes how each speech token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = (
            torch.cumsum((special_speech_token_mask * (speech_len - 1) + 1), -1) - 1
        )  # (N,U)
        nb_speech_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_speech_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_speech_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim),
                IGNORE_TOKEN_ID,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_speech_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_speech_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<speech>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the speech features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_speech_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_speech_indices
        ]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_speech_indices
            ]

        # 5. Fill the embeddings corresponding to the speechs. Anything that is not `text_positions` needs filling (#29835)
        speech_to_overwrite = torch.full(
            (batch_size, max_embed_dim),
            True,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        speech_to_overwrite[batch_indices, text_to_overwrite] = False
        if speech_lens is not None:
            speech_pad_position = speech_to_overwrite.cumsum(-1) <= speech_lens[:, None]
        speech_to_overwrite &= speech_to_overwrite.cumsum(-1) - 1 >= nb_speech_pad[
            :, None
        ].to(target_device)

        if speech_to_overwrite.sum() != speech_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of speech tokens is {torch.sum(special_speech_token_mask)} while"
                f" the number of speech given to the model is {num_speechs}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[speech_to_overwrite] = (
            speech_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        if speech_lens is not None:
            speech_to_overwrite &= speech_pad_position
        final_attention_mask |= speech_to_overwrite

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(
            input_ids == self.llm.config.pad_token_id
        )
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels #, position_ids


from pathlib import Path
import openvino as ov
from openvino import save_model, convert_model
import gc
from transformers import AutoConfig
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
import numpy as np
try:
    from openvino import opset13
except ImportError:
    from openvino.runtime import opset13

def model_has_state(ov_model: ov.Model):
    return len(ov_model.get_sinks()) > 0

def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])

def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: list[str],
    key_value_input_names: list[str],
    gather_dim: int,
):
    """
    Fuses reored_cache during generate cycle into ov.Model. Used with stateful models, because we can not modify model state directly.

    Adds a new beam_idx parameter and Gather op per each kv-cache input in a given model.
    Should be run before make_stateful. Implements optimumum's _reorder_cache
    inside the model in the beginning of each iteration.
    Gather works along given gather_dim dimension that may vary from model to model.
    KV-cache inputs are identified based on names in key_value_input_names.
    Append the new beam_idx parameter to not_kv_inputs.

    Parameters:
      ov_model (`ov.Model`):
          openvino model for processing
      not_kv_inputs (`list[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`list[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()

def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
    input_ids = ov_model.input("inputs_embeds")
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()

def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: list[str],
    key_value_input_names: list[str],
    key_value_output_names: list[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`list[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`list[str]`):
            list of names for key value input layers
        key_value_output_names (`list[str]`):
            list of names for key value input layers
        batch_dim (int):
            index of batch dimension in key value layers
        num_attention_heads (int):
            number of attention heads for batch dimension initialization
        num_beams_an_batch (int):
            precalculated number of beams and batch for shapes initialization
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)

def patch_stateful(ov_model, input_num = 2, output_num = 1):
    # key_value_input_names = [key.get_any_name() for key in ov_model.inputs[input_num:]]
    # key_value_output_names = [key.get_any_name() for key in ov_model.outputs[output_num:]]
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs if any("key_values" in key_name for key_name in key.get_names())]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs if any("present" in key_name for key_name in key.get_names())]
    not_kv_inputs = [input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())]
    # print(f"### key_value_input_names: {[input for input in key_value_input_names]}")
    # print(f"### key_value_output_names: {[input for input in key_value_output_names]}")
    # print(f"### not_kv_inputs: {[input.get_any_name() for input in not_kv_inputs]}")
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 0
    num_attention_heads = 1

    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    make_stateful(
        ov_model,
        not_kv_inputs,
        key_value_input_names,
        key_value_output_names,
        batch_dim,
        num_attention_heads,
        None,
    )
    
def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()
    gc.collect()


FireRedAsrLLM_Encoder_MODEL_NAME = "FireRedASR_llm_encoder_ov.xml"
FireRedAsrLLM_LLM_EMBED_MODEL_NAME = "Qwen2-7B-Instruct_embed_tokens.xml"
FireRedAsrLLM_LLM_DECODER_MODEL_NAME = "Qwen2-7B-Instruct_decoder.xml"
FireRedAsrLLM_LLM_TOKENIZER_MODEL_NAME = "Qwen2-7B-Instruct_tokenizer.xml"

class FireRedAsr_LLM_ov(GenerationMixin) :
    def __init__(self, ov_core, model_path, infer_type):
        self.request = None
        self._past_length = 0
        self.config = None
        self.device = torch.device("cpu")
        self._supports_cache_class = False
        self.converted_to_ov = False
        try :
            if ov_core is None :
                ov_core = ov.Core()
            ov_path = Path(model_path)
            self.ov_path_llm = ov_path.parent / "ov_model" / FireRedAsrLLM_LLM_DECODER_MODEL_NAME
            self.ov_path_llm_embed = ov_path.parent / "ov_model" / FireRedAsrLLM_LLM_EMBED_MODEL_NAME
            self.ov_path_llm_config = ov_path.parent / "ov_model/config.json"
            if not self.ov_path_llm.exists() or not self.ov_path_llm_embed.exists() or not self.ov_path_llm_config.exists():
                self.converted_to_ov = True
            self.ov_dtype = infer_type
            ov_config = {'INFERENCE_PRECISION_HINT':self.ov_dtype, 'PERFORMANCE_HINT': "LATENCY"}
            self.ov_embed_model = ov_core.compile_model(self.ov_path_llm_embed, 'CPU', ov_config)
            self.embed_request = self.ov_embed_model.create_infer_request()
            
            self.config = AutoConfig.from_pretrained(self.ov_path_llm_config)
            self.generation_config = GenerationConfig.from_model_config(self.config)
            
            self.ov_llm_model = ov_core.compile_model(self.ov_path_llm, 'CPU', ov_config)
            self.request = self.ov_llm_model.create_infer_request()
            
            self.using_ov = True
            self.main_input_name = "input_ids"
            self.input_names = [input_t.get_any_name() for input_t in self.ov_llm_model.inputs]
        except Exception as e:
            self.using_ov = False
            print(f"### ov load {self.ov_path_llm} or {self.ov_path_llm_embed} or {self.ov_path_llm_config} "
                  f"failed.\nError Message: {e}")

    def forward(self, input_ids, attention_mask = None, past_key_values = None,
                position_ids = None, inputs_embeds = None, **kwargs, ):
        inputs = self.prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        
        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        logits = torch.from_numpy(self.request.get_tensor("logits").data).float()
        past_key_values = ((),)
        self._past_length += inputs["inputs_embeds"].shape[1]
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)
    
    def prepare_inputs(self, input_ids: torch.LongTensor, attention_mask = None,
                       past_key_values = None, position_ids = None, inputs_embeds = None, **kwargs,):
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        if inputs_embeds is None:
            self.embed_request.start_async(input_ids if past_key_values is None else input_ids[:, -1:])

        inputs = {}
        if past_key_values is None:
            if self.request is not None:
                self.request.reset_state()
                self.next_beam_idx = np.arange(batch_size, dtype=int)
                self._past_length = 0
        past_len = self._get_past_length(past_key_values)

        if "attention_mask" in self.input_names or "position_ids" in self.input_names:
            if attention_mask is not None:
                attention_mask = np.array(attention_mask)
            else:
                attention_mask = np.ones((inputs_embeds.shape[0], inputs_embeds.shape[1] + past_len), dtype=int)

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "position_ids" in self.input_names:
            if position_ids is not None:
                position_ids = np.array(position_ids)
            else:
                position_ids = np.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

            inputs["position_ids"] = position_ids

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)

        if inputs_embeds is None:
            self.embed_request.wait()
            inputs_embeds = self.embed_request.get_output_tensor(0)
            # print(f"### inputs_embeds: {inputs_embeds}")

            if hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb

        inputs["inputs_embeds"] = inputs_embeds

        return inputs
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        if past_key_values is not None:
            past_len = self._get_past_length(past_key_values)
            if attention_mask is not None and input_ids is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_len) :]
            elif input_ids is not None and past_len < input_ids.shape[1]:
                input_ids = input_ids[:, past_len:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None and "position_ids" in self.input_names:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values and input_ids is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds if past_key_values is None else None,
        }

        return model_inputs

    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        return self._past_length

    def _reorder_cache(self, past_key_values: tuple[tuple[torch.Tensor]], beam_idx: torch.Tensor) -> tuple[tuple[torch.Tensor]]:
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def can_generate(self):
        return True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class FireRedAsr_LLM_ov_wrapper :
    def __init__(self, ov_core, model_path, infer_type) :
        self.llm = FireRedAsr_LLM_ov(ov_core, model_path, infer_type)

class FireRedAsrLlm_ov :
    def __init__(self, ov_core, model_path, infer_type):
        self.converted_to_ov = False
        self.ov_encoder_model = None
        self.llm_model = None
        
        try :
            if ov_core is None :
                ov_core = ov.Core()
            ov_path = Path(model_path)
            self.ov_path_encoder = ov_path.parent / "ov_model" / FireRedAsrLLM_Encoder_MODEL_NAME
            if not self.ov_path_encoder.exists():
                self.converted_to_ov = True
            self.ov_dtype = infer_type
            ov_config = {'INFERENCE_PRECISION_HINT':self.ov_dtype, 'PERFORMANCE_HINT': "LATENCY"}
            self.ov_encoder_model = ov_core.compile_model(self.ov_path_encoder, 'CPU', ov_config)
            self.llm_model = FireRedAsr_LLM_ov_wrapper(ov_core, model_path, infer_type)
            self.using_ov = True
            if self.llm_model.llm.converted_to_ov :
                self.converted_to_ov = True
            self.using_ov = self.llm_model.llm.using_ov
        except Exception as e:
            self.using_ov = False
            print(f"### ov load {self.ov_path_encoder} failed.\nError Message: {e}")
            
        
    @torch.inference_mode()
    def transcribe(self, padded_feat, feat_lengths, padded_input_ids, attention_mask,
                   beam_size=1, decode_max_len=0, decode_min_len=0,
                   repetition_penalty=1.0, llm_length_penalty=1.0, temperature=1.0):
        if self.ov_encoder_model :
            res = self.ov_encoder_model((padded_feat, feat_lengths, padded_input_ids))
            inputs_embeds = torch.from_numpy(res[0])
            speech_features = torch.from_numpy(res[1])
            speech_lens = torch.from_numpy(res[2])
        else :
            encoder_outs, enc_lengths, enc_mask = self.llm_model.encoder(padded_feat, feat_lengths)
            speech_features, speech_lens = self.llm_model.encoder_projector(encoder_outs, enc_lengths)
            inputs_embeds = self.llm_model.llm.get_input_embeddings()(padded_input_ids)
        
            if self.converted_to_ov :
                self.convert_ov_model_encoder(padded_feat, feat_lengths, padded_input_ids)

        inputs_embeds, attention_mask, _ = \
            self._merge_input_ids_with_speech_features(
                speech_features.to(inputs_embeds.dtype), inputs_embeds, padded_input_ids, attention_mask,
                speech_lens=speech_lens
            )

        max_new_tokens = speech_features.size(1) if decode_max_len < 1 else decode_max_len
        max_new_tokens = max(1, max_new_tokens)

        if self.converted_to_ov :
            self.convert_qwen2llm_model()
            exit(0)

        generated_ids = self.llm_model.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            do_sample=False,
            min_length=decode_min_len,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            length_penalty=llm_length_penalty,
            temperature=temperature,
            bos_token_id=self.llm_model.llm.config.bos_token_id,
            eos_token_id=self.llm_model.llm.config.eos_token_id,
            pad_token_id=self.llm_model.llm.config.pad_token_id,
        )

        return generated_ids


    def _merge_input_ids_with_speech_features(
        self, speech_features, inputs_embeds, input_ids, attention_mask, labels=None,
        speech_lens=None
    ):
        """
        Modified from: https://github.com/k2-fsa/icefall/blob/master/egs/speech_llm/ASR_LLM/whisper_llm_zh/model.py
        """
        speech_lens = None
        num_speechs, speech_len, embed_dim = speech_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(self.llm_model.llm.config.pad_token_id)
        )
        # 1. Create a mask to know where special speech tokens are
        special_speech_token_mask = input_ids == self.llm_model.llm.config.default_speech_token_id
        num_special_speech_tokens = torch.sum(special_speech_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (
            num_special_speech_tokens.max() * (speech_len - 1)
        ) + sequence_length
        batch_indices, non_speech_indices = torch.where(
            input_ids != self.llm_model.llm.config.default_speech_token_id
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged speech-text sequence.
        # `special_speech_token_mask` identifies speech tokens. Each speech token will be replaced by `nb_text_tokens_per_speechs - 1` text tokens.
        # `torch.cumsum` computes how each speech token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = (
            torch.cumsum((special_speech_token_mask * (speech_len - 1) + 1), -1) - 1
        )  # (N,U)
        nb_speech_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_speech_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_speech_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim),
                IGNORE_TOKEN_ID,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_speech_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_speech_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<speech>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the speech features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_speech_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_speech_indices
        ]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_speech_indices
            ]

        # 5. Fill the embeddings corresponding to the speechs. Anything that is not `text_positions` needs filling (#29835)
        speech_to_overwrite = torch.full(
            (batch_size, max_embed_dim),
            True,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        speech_to_overwrite[batch_indices, text_to_overwrite] = False
        if speech_lens is not None:
            speech_pad_position = speech_to_overwrite.cumsum(-1) <= speech_lens[:, None]
        speech_to_overwrite &= speech_to_overwrite.cumsum(-1) - 1 >= nb_speech_pad[
            :, None
        ].to(target_device)

        if speech_to_overwrite.sum() != speech_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of speech tokens is {torch.sum(special_speech_token_mask)} while"
                f" the number of speech given to the model is {num_speechs}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[speech_to_overwrite] = (
            speech_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        if speech_lens is not None:
            speech_to_overwrite &= speech_pad_position
        final_attention_mask |= speech_to_overwrite

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(
            input_ids == self.llm_model.llm.config.pad_token_id
        )
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels #, position_ids


    def eval(self):
        return self


    def cpu(self):
        return self


    def convert_ov_model_encoder(self, padded_feat, feat_lengths, padded_input_ids):
        class ModelEncodeWrapper(torch.nn.Module):
            def __init__(self, encoder, encoder_projector, llm):
                super().__init__()
                self.encoder = encoder
                self.encoder_projector = encoder_projector
                self.llm = llm

            def forward(self, padded_feat, feat_lengths, padded_input_ids):
                encoder_outs, enc_lengths, enc_mask = self.encoder(padded_feat, feat_lengths)
                speech_features, speech_lens = self.encoder_projector(encoder_outs, enc_lengths)
                inputs_embeds = self.llm.get_input_embeddings()(padded_input_ids)
                return inputs_embeds, speech_features, speech_lens

        model = ModelEncodeWrapper(self.llm_model.encoder, self.llm_model.encoder_projector, self.llm_model.llm)
        model.eval()
        example_inputs = {"padded_feat":padded_feat, "feat_lengths":feat_lengths, "padded_input_ids":padded_input_ids}
        ov_model = convert_model(model, example_input=example_inputs)
        save_model(ov_model, self.ov_path_encoder)


    def convert_qwen2llm_model(self, quantization_config=None):
        ov_path_llm = self.ov_path_encoder.parent / FireRedAsrLLM_LLM_DECODER_MODEL_NAME
        ov_path_llm_embed = self.ov_path_encoder.parent / FireRedAsrLLM_LLM_EMBED_MODEL_NAME
        ov_path_llm_config = self.ov_path_encoder.parent

        if not ov_path_llm_embed.exists():
            print("⌛ Convert Input embedding model")
            input_ids = torch.tensor([[111268],[ 18493],[ 87026]])
            ov_model = ov.convert_model(
                self.llm_model.llm.model.embed_tokens,
                example_input=input_ids,
            )
            ov.save_model(ov_model, ov_path_llm_embed)
            del ov_model
            cleanup_torchscript_cache()
            print(f"✅ Input embedding model successfully converted to {ov_path_llm_embed}")

        if not ov_path_llm.exists():
            print("⌛ Convert Language model")
            self.llm_model.llm.model.config.save_pretrained(ov_path_llm_config)
            from transformers.cache_utils import DynamicCache
            class llm_wrapper(torch.nn.Module):
                def __init__(self, qwen2lm):
                    super().__init__()
                    self.qwen2lm = qwen2lm
                
                def forward(self, inputs_embeds, attention_mask, past_key_values):
                    with torch.no_grad():
                        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                        result = self.qwen2lm(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            # position_ids=position_ids,
                            output_hidden_states=True,
                            return_dict=True,
                            use_cache=True,
                            past_key_values=past_key_values,
                        )
                        return (result.logits, result.past_key_values.to_legacy_cache())

            lang_model = llm_wrapper(self.llm_model.llm)
            lang_model.eval()
            
            hidden_size = self.llm_model.llm.config.hidden_size
            num_pkv = self.llm_model.llm.config.num_hidden_layers
            num_key_value_heads = self.llm_model.llm.config.num_key_value_heads
            num_attention_heads = self.llm_model.llm.config.num_attention_heads
            num_tokens = 66
            pkv_shape = (3, num_key_value_heads, num_tokens, hidden_size // num_attention_heads)
            inputs_embeds = torch.randn((3, 1, hidden_size))
            attention_mask = torch.ones([3, num_tokens+1], dtype=torch.long)
            # position_ids = torch.tensor([[num_tokens], [num_tokens], [num_tokens]], dtype=torch.long)
            print(f"hidden_size={hidden_size}, num_pkv={num_pkv}, pkv_shape={pkv_shape}, "
                  f"inputs_embeds={inputs_embeds.shape}, attention_mask={attention_mask.shape}, ")
                #   f"position_ids={position_ids.shape}")
        
            input_names = ["inputs_embeds", "attention_mask"]#, "position_ids"]#, "cache_position"]
            output_names = ["logits"]
            past_key_values = []
            for i in range(num_pkv):
                kv = [torch.randn(pkv_shape) for _ in range(2)]
                past_key_values.append(kv)
                input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
                output_names.extend([f"present.{i}.key", f"present.{i}.value"])

            example_input = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, "past_key_values": past_key_values}
                            #   "position_ids": position_ids, }
            
            ov_model = ov.convert_model(lang_model, example_input=example_input)

            for input, input_name in zip(ov_model.inputs, input_names):
                input.get_tensor().set_names({input_name})

            for output, output_name in zip(ov_model.outputs, output_names):
                output.get_tensor().set_names({output_name})

            patch_stateful(ov_model, 2, 1)
            print("✅ Language model successfully converted")

            if quantization_config is not None:
                print(f"⌛ Weights compression with {quantization_config['mode']} mode started")
                ov_model = nncf.compress_weights(ov_model, **quantization_config)
                print("✅ Weights compression finished")
            else :
                ov_model.set_rt_info("f16", ["runtime_options", "KV_CACHE_PRECISION"])

            ov.save_model(ov_model, ov_path_llm)
            del ov_model
            cleanup_torchscript_cache()
            print(f"✅ model conversion finished. You can find results in {ov_path_llm}")


