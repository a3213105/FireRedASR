import torch
import time
from fireredasr.models.module.conformer_encoder import ConformerEncoder
from fireredasr.models.module.transformer_decoder import TransformerDecoder
import json

class FireRedAsrAed(torch.nn.Module):
    @classmethod
    def from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super().__init__()
        self.sos_id = args.sos_id
        self.eos_id = args.eos_id

        self.encoder = ConformerEncoder(
            args.idim, args.n_layers_enc, args.n_head, args.d_model,
            args.residual_dropout, args.dropout_rate,
            args.kernel_size, args.pe_maxlen)

        self.decoder = TransformerDecoder(
            args.sos_id, args.eos_id, args.pad_id, args.odim,
            args.n_layers_dec, args.n_head, args.d_model,
            args.residual_dropout, args.pe_maxlen)

    def transcribe(self, padded_input, input_lengths,
                   beam_size=1, nbest=1, decode_max_len=0,
                   softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0):
        enc_outputs, _, enc_mask = self.encoder(padded_input, input_lengths)
        nbest_hyps = self.decoder.batch_beam_search(
            enc_outputs, enc_mask,
            beam_size, nbest, decode_max_len,
            softmax_smoothing, length_penalty, eos_penalty)
        return nbest_hyps
    
    def encode(self, padded_input, input_lengths):
        enc_outputs, _, enc_mask = self.encoder(padded_input, input_lengths)
        return enc_outputs, enc_mask

from pathlib import Path
import openvino as ov
from openvino import save_model, convert_model
import gc
from transformers import AutoConfig
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
import numpy as np
import torch 
import torch.nn.functional as F

try:
    from openvino import opset13
except ImportError:
    from openvino.runtime import opset13

def model_has_state(ov_model: ov.Model):
    return len(ov_model.get_sinks()) > 0

def model_has_input_output_name(ov_model: ov.Model, name: str):
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])

def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: list[str],
    key_value_input_names: list[str],
    gather_dim: int,
):
    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("encoder_outputs").get_partial_shape()[0]
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
    input_ids = ov_model.input("encoder_outputs")
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
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs if any("key_values" in key_name for key_name in key.get_names())]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs if any("present" in key_name for key_name in key.get_names())]
    not_kv_inputs = [input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())]
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
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()
    gc.collect()

FireRedAsrAed_CONFIG_NAME = "FireRedASR_AED_config.json"
FireRedAsrAed_MODEL_NAME = "FireRedASR_AED_ov.xml"
FireRedAsrAed_Encoder_MODEL_NAME = "FireRedASR_AED_encoder_ov.xml"
FireRedAsrAed_Decoder_MODEL_NAME = "FireRedASR_AED_decoder_ov.xml"
FireRedAsrAed_Decoder0_MODEL_NAME = "FireRedASR_AED_decoder0_ov.xml"
FireRedAsrAed_Decoder1_MODEL_NAME = "FireRedASR_AED_decoder1_ov.xml"
class FireRedAsrAed_ov :
    def __init__(self, args, ov_core, model_path, enc_type, dec_type, cache_size):
        ov_path = Path(model_path)
        self.ov_config_path = ov_path.parent / "ov_model" / FireRedAsrAed_CONFIG_NAME
        if args is None :
            self.load_config()
        else :
            self.sos_id = args.sos_id
            self.eos_id = args.eos_id
            self.pad_id = args.pad_id
        # self.n_layers = args.n_layers_dec
        self.INF = 1e10
        self.next_beam_idx = None
        self.infer_mode = 0
        
        self.torch_model = None
        self.converted_to_ov = False
        self.using_ov = False

        self.ov_core = ov_core
        self.ov_encoder_path = ov_path.parent / "ov_model" / FireRedAsrAed_Encoder_MODEL_NAME
        self.ov_decoder0_path = ov_path.parent / "ov_model" / FireRedAsrAed_Decoder0_MODEL_NAME
        self.ov_decoder1_path = ov_path.parent / "ov_model" / FireRedAsrAed_Decoder1_MODEL_NAME
        if not self.ov_encoder_path.exists() or not self.ov_decoder0_path.exists() or not self.ov_decoder1_path.exists():
            self.converted_to_ov = True
        self.enc_type = enc_type
        self.dec_type = dec_type
        if self.enc_type in "f32f16bf16" and self.dec_type in "f32f16bf16" :
            self.load_ov_model(cache_size)

    def load_ov_model(self, cache_size):
        try :
            if self.ov_core is None :
                self.ov_core = ov.Core()
            cache_size_str = f"{cache_size}"
            print(f"cache_size_str={cache_size_str}")
            self.ov_core.set_property("CPU", {"CPU_RUNTIME_CACHE_CAPACITY": cache_size_str})
            # self.ov_core.set_property("CPU", {"ENABLE_MMAP": False})
            ov_config = {'INFERENCE_PRECISION_HINT': self.enc_type, 'PERFORMANCE_HINT': "LATENCY"}
            self.ov_encoder_model = self.ov_core.compile_model(self.ov_encoder_path, 'CPU', ov_config)
            ov_config = {'INFERENCE_PRECISION_HINT': self.dec_type, 'PERFORMANCE_HINT': "LATENCY"}
            self.ov_decoder0_model = self.ov_core.compile_model(self.ov_decoder0_path, 'CPU', ov_config)
            self.ov_decoder1_model = self.ov_core.compile_model(self.ov_decoder1_path, 'CPU', ov_config)
            self.using_ov = True
        except Exception as e:
            print(f"### ov load {self.ov_encoder_path} or {self.ov_decoder0_path} or {self.ov_decoder1_path} failed, {e}")

    def get_ys_lengths(self, ys):
        N, B, Tmax = ys.size()
        ys_lengths = torch.sum(torch.ne(ys, self.eos_id), dim=-1)
        return ys_lengths.int()

    def batch_beam_search_for0(self, ys, tgt_mask, encoder_outputs, src_mask) :
        dec_output = self.torch_model.decoder.tgt_word_emb(ys) * self.torch_model.decoder.scale + self.torch_model.decoder.positional_encoding(ys)
        new_caches: List[Optional[Tensor]] = []
        for i, dec_layer in enumerate(self.torch_model.decoder.layer_stack):
            dec_output = dec_layer.forward0(
                dec_output, encoder_outputs,
                tgt_mask, src_mask)
            # caches[i] = dec_output
            new_caches.append(dec_output)

        dec_output = self.torch_model.decoder.layer_norm_out(dec_output)
        t_logit = self.torch_model.decoder.tgt_word_prj(dec_output[:, -1])
        return t_logit, new_caches

    def batch_beam_search_for1(self, ys, tgt_mask, encoder_outputs, src_mask, caches) :
        dec_output = self.torch_model.decoder.tgt_word_emb(ys) * self.torch_model.decoder.scale + self.torch_model.decoder.positional_encoding(ys)
        new_caches: List[Optional[Tensor]] = []
        for i, dec_layer in enumerate(self.torch_model.decoder.layer_stack):
            dec_output = dec_layer.forward1(
                dec_output, encoder_outputs,
                tgt_mask, src_mask, caches[i])
            # caches[i] = dec_output
            new_caches.append(dec_output)

        dec_output = self.torch_model.decoder.layer_norm_out(dec_output)
        t_logit = self.torch_model.decoder.tgt_word_prj(dec_output[:, -1])
        # print(f"batch_beam_search_for1 out:{t_logit.shape}, new_caches={len(new_caches)}, new_caches0={new_caches[0].shape}")
        return t_logit, new_caches

    def batch_beam_search(self, encoder_outputs, src_masks,
                   beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):
        B = beam_size
        N, Ti, H = encoder_outputs.size()
        device = encoder_outputs.device
        maxlen = decode_max_len if decode_max_len > 0 else Ti
        assert eos_penalty > 0.0 and eos_penalty <= 1.0

        # Init
        encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, Ti, H)
        src_mask = src_masks.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, -1, Ti)
        ys = torch.ones(N*B, 1).fill_(self.sos_id).long().to(device)
        # caches: List[Optional[Tensor]] = []
        # for _ in range(self.n_layers):
        #     caches.append(None)
        scores = torch.tensor([0.0] + [-self.INF]*(B-1)).float().to(device)
        scores = scores.repeat(N).view(N*B, 1)
        is_finished = torch.zeros_like(scores)

        #round 0
        tgt_mask = self.torch_model.decoder.ignored_target_position_is_0(ys, self.pad_id)
        
        t_logit, caches = self.batch_beam_search_for0(ys, tgt_mask, encoder_outputs, src_mask)
        # print(f"softmax_smoothing:{softmax_smoothing}, eos_penalty:{eos_penalty}")
        t_scores = F.log_softmax(t_logit / softmax_smoothing, dim=-1)

        if eos_penalty != 1.0:
            t_scores[:, self.eos_id] *= eos_penalty

        t_topB_scores, t_topB_ys = torch.topk(t_scores, k=B, dim=1)
        t_topB_scores = self.torch_model.decoder.set_finished_beam_score_to_zero(t_topB_scores, is_finished)
        t_topB_ys = self.torch_model.decoder.set_finished_beam_y_to_eos(t_topB_ys, is_finished)

        # Accumulated
        scores = scores + t_topB_scores

        # Pruning
        scores = scores.view(N, B*B)
        scores, topB_score_ids = torch.topk(scores, k=B, dim=1)
        scores = scores.view(-1, 1)

        topB_row_number_in_each_B_rows_of_ys = torch.div(topB_score_ids, B).view(N*B)
        stride = B * torch.arange(N).view(N, 1).repeat(1, B).view(N*B).to(device)
        topB_row_number_in_ys = topB_row_number_in_each_B_rows_of_ys.long() + stride.long()

        # Update ys
        ys = ys[topB_row_number_in_ys]
        t_ys = torch.gather(t_topB_ys.view(N, B*B), dim=1, index=topB_score_ids).view(N*B, 1)
        ys = torch.cat((ys, t_ys), dim=1)

        # Update caches
        new_caches: List[Optional[Tensor]] = []
        for cache in caches:
            if cache is not None:
                new_caches.append(cache[topB_row_number_in_ys])
        caches = new_caches

        # Update finished state
        is_finished = t_ys.eq(self.eos_id)

        if is_finished.sum().item() != N*B:
            # Autoregressive Prediction
            for t in range(1, maxlen):
                tgt_mask = self.torch_model.decoder.ignored_target_position_is_0(ys, self.pad_id)

                t_logit, caches = self.batch_beam_search_for1(ys, tgt_mask, encoder_outputs, src_mask, caches)
                t_scores = F.log_softmax(t_logit / softmax_smoothing, dim=-1)

                if eos_penalty != 1.0:
                    t_scores[:, self.eos_id] *= eos_penalty

                t_topB_scores, t_topB_ys = torch.topk(t_scores, k=B, dim=1)
                t_topB_scores = self.torch_model.decoder.set_finished_beam_score_to_zero(t_topB_scores, is_finished)
                t_topB_ys = self.torch_model.decoder.set_finished_beam_y_to_eos(t_topB_ys, is_finished)

                # Accumulated
                scores = scores + t_topB_scores

                # Pruning
                scores = scores.view(N, B*B)
                scores, topB_score_ids = torch.topk(scores, k=B, dim=1)
                scores = scores.view(-1, 1)

                topB_row_number_in_each_B_rows_of_ys = torch.div(topB_score_ids, B).view(N*B)
                stride = B * torch.arange(N).view(N, 1).repeat(1, B).view(N*B).to(device)
                topB_row_number_in_ys = topB_row_number_in_each_B_rows_of_ys.long() + stride.long()

                # Update ys
                ys = ys[topB_row_number_in_ys]
                t_ys = torch.gather(t_topB_ys.view(N, B*B), dim=1, index=topB_score_ids).view(N*B, 1)
                ys = torch.cat((ys, t_ys), dim=1)

                # Update caches
                new_caches: List[Optional[Tensor]] = []
                for cache in caches:
                    if cache is not None:
                        new_caches.append(cache[topB_row_number_in_ys])
                caches = new_caches

                # Update finished state
                is_finished = t_ys.eq(self.eos_id)
                if is_finished.sum().item() == N*B:
                    break

        # Length penalty (follow GNMT)
        scores = scores.view(N, B)
        ys = ys.view(N, B, -1)
        ys_lengths = self.get_ys_lengths(ys)
        if length_penalty > 0.0:
            penalty = torch.pow((5+ys_lengths.float())/(5.0+1), length_penalty)
            scores /= penalty
        nbest_scores, nbest_ids = torch.topk(scores, k=int(nbest), dim=1)
        nbest_scores = -1.0 * nbest_scores
        index = nbest_ids + B * torch.arange(N).view(N, 1).to(device).long()
        nbest_ys = ys.view(N*B, -1)[index.view(-1)]
        nbest_ys = nbest_ys.view(N, nbest_ids.size(1), -1)
        nbest_ys_lengths = ys_lengths.view(N*B)[index.view(-1)].view(N, -1)

        # result
        nbest_hyps: List[List[Dict[str, Tensor]]] = []
        for n in range(N):
            n_nbest_hyps: List[Dict[str, Tensor]] = []
            for i, score in enumerate(nbest_scores[n]):
                new_hyp = {
                    "yseq": nbest_ys[n, i, 1:nbest_ys_lengths[n, i]]
                }
                n_nbest_hyps.append(new_hyp)
            nbest_hyps.append(n_nbest_hyps)
        return nbest_hyps
    
    def batch_beam_search_for0_ov(self, ys, encoder_outputs, src_mask, scores, is_finished,
                                  softmax_smoothing, eos_penalty, B, N) :
        res = self.ov_decoder0_model((ys, encoder_outputs, src_mask, scores, is_finished, softmax_smoothing, eos_penalty, B, N),
                                     share_inputs = False)
        # print(f"res = {len(res)}, {res}")
        # t_scores = torch.from_numpy(res[0])
        new_caches = []
        for i in range(3, len(res)):
            new_caches.append(res[i])
        return res[0], res[1], res[2], new_caches

    def batch_beam_search_for1_ov(self, ys, encoder_outputs, src_mask, scores, is_finished,
                                  softmax_smoothing, eos_penalty, B, N, caches) :
        res = self.ov_decoder1_model((ys, encoder_outputs, src_mask, scores, is_finished, softmax_smoothing, eos_penalty, B, N, *caches),
                                     share_inputs = False)
        new_caches = []
        for i in range(3, len(res)):
            new_caches.append(res[i])
        return res[0], res[1], res[2], new_caches

    def batch_beam_search1(self, encoder_outputs, src_masks,
                   beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):
        B = beam_size
        N, Ti, H = encoder_outputs.size()
        device = encoder_outputs.device
        maxlen = decode_max_len if decode_max_len > 0 else Ti
        assert eos_penalty > 0.0 and eos_penalty <= 1.0

        # Init
        encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, Ti, H)
        src_mask = src_masks.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, -1, Ti)
        ys = torch.ones(N*B, 1).fill_(self.sos_id).long().to(device)
        scores = torch.tensor([0.0] + [-self.INF]*(B-1)).float().to(device)
        scores = scores.repeat(N).view(N*B, 1)
        is_finished = torch.zeros_like(scores)

        t_ys, scores, ys, caches = self.batch_beam_search_for0_ov(ys, encoder_outputs,
                                            src_mask, scores, is_finished, softmax_smoothing,
                                            eos_penalty, B, N)
        t_ys = torch.from_numpy(t_ys)
        # Update finished state
        is_finished = t_ys.eq(self.eos_id)
        if is_finished.sum().item() != N*B:
            # Autoregressive Prediction
            for t in range(1, maxlen):
                t_ys, scores, ys, caches = self.batch_beam_search_for1_ov(ys, encoder_outputs, src_mask, scores, is_finished,
                                                             softmax_smoothing, eos_penalty, B, N, caches)
                t_ys = torch.from_numpy(t_ys)
                # Update finished state
                is_finished = t_ys.eq(self.eos_id)
                if is_finished.sum().item() == N*B:
                    break

        scores = torch.from_numpy(scores)
        ys = torch.from_numpy(ys)
        # Length penalty (follow GNMT)
        scores = scores.view(N, B)
        ys = ys.view(N, B, -1)
        ys_lengths = self.get_ys_lengths(ys)
        if length_penalty > 0.0:
            penalty = torch.pow((5+ys_lengths.float())/(5.0+1), length_penalty)
            scores /= penalty
        nbest_scores, nbest_ids = torch.topk(scores, k=int(nbest), dim=1)
        nbest_scores = -1.0 * nbest_scores
        index = nbest_ids + B * torch.arange(N).view(N, 1).to(device).long()
        nbest_ys = ys.view(N*B, -1)[index.view(-1)]
        nbest_ys = nbest_ys.view(N, nbest_ids.size(1), -1)
        nbest_ys_lengths = ys_lengths.view(N*B)[index.view(-1)].view(N, -1)

        # result
        nbest_hyps: List[List[Dict[str, Tensor]]] = []
        for n in range(N):
            n_nbest_hyps: List[Dict[str, Tensor]] = []
            for i, score in enumerate(nbest_scores[n]):
                new_hyp = {
                    "yseq": nbest_ys[n, i, 1:nbest_ys_lengths[n, i]]
                }
                n_nbest_hyps.append(new_hyp)
            nbest_hyps.append(n_nbest_hyps)
        return nbest_hyps

    @torch.inference_mode()
    def transcribe0(self, padded_input, input_lengths,
                   beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):
        # if self.converted_to_ov :
        #     self.convert_ov_model(padded_input, input_lengths, beam_size, nbest, decode_max_len,
        #         softmax_smoothing, length_penalty, eos_penalty)
        #     self.load_ov_model()

        # print(f"inputs shape: padded_input:{padded_input.shape}, input_lengths:{input_lengths.shape}")
        inputs = (padded_input, input_lengths)
        res = self.ov_encoder_model(inputs)
        enc_outputs = torch.from_numpy(res[0])
        enc_mask = torch.from_numpy(res[1])
        nbest_hyps = self.batch_beam_search1(enc_outputs, enc_mask, beam_size, nbest,
                                decode_max_len, softmax_smoothing, length_penalty, eos_penalty)
        return nbest_hyps

    @torch.inference_mode()
    def transcribe(self, padded_input, input_lengths,
                   beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):
        if self.torch_model is None :
            return self.transcribe0(padded_input, input_lengths,
                   beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty)
        else :
            if self.converted_to_ov :
                self.convert_ov_model(padded_input, input_lengths, beam_size, nbest, decode_max_len,
                    softmax_smoothing, length_penalty, eos_penalty)

            return self.torch_model.transcribe(padded_input, input_lengths,
                   beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty)

            # enc_outputs, enc_mask = self.torch_model.encode(padded_input, input_lengths)
            # nbest_hyps = self.batch_beam_search(enc_outputs, enc_mask, beam_size, nbest,
            #         decode_max_len, softmax_smoothing, length_penalty, eos_penalty)
            # return nbest_hyps

    @torch.inference_mode()
    def transcribe1(self, padded_input, input_lengths,
                   beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):
        if self.converted_to_ov :
            self.convert_ov_model(padded_input, input_lengths, beam_size, nbest, decode_max_len,
                softmax_smoothing, length_penalty, eos_penalty)

        if self.infer_mode == 0 :
            enc_outputs, enc_mask = self.torch_model.encode(padded_input, input_lengths)
            nbest_hyps = self.batch_beam_search(enc_outputs, enc_mask, beam_size, nbest,
                    decode_max_len, softmax_smoothing, length_penalty, eos_penalty)
        else :
            if self.infer_mode < 3 :
                inputs = (padded_input, input_lengths)
                res = self.ov_encoder_model(inputs)
                enc_outputs = torch.from_numpy(res[0])
                enc_mask = torch.from_numpy(res[1])
            else :
                enc_outputs, enc_mask = self.torch_model.encode(padded_input, input_lengths)
            st1 = time.perf_counter()
            if self.infer_mode > 1:
                nbest_hyps = self.batch_beam_search1(enc_outputs, enc_mask, beam_size, nbest,
                                        decode_max_len, softmax_smoothing, length_penalty, eos_penalty)
            else :
                nbest_hyps = self.batch_beam_search(enc_outputs, enc_mask, beam_size, nbest,
                                        decode_max_len, softmax_smoothing, length_penalty, eos_penalty)
        return nbest_hyps

    def eval(self):
        if self.torch_model is not None :
            self.torch_model.eval()  

    def cpu(self):
        if self.torch_model is not None :
            self.torch_model.cpu()  

    def load_config(self):
        # print(f"Load model config from {self.ov_config_path}")
        with open(self.ov_config_path, 'r') as file:
            data = json.load(file)
            self.sos_id = data["sos_id"]
            self.eos_id = data["eos_id"]
            self.pad_id = data["pad_id"]
            return True
        return False

    @torch.inference_mode()
    def convert_ov_model(self, feats, lengths, beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):  
        class ModelEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.eval()

            def forward(self, feats, lengths):
                with torch.no_grad():
                    enc_outputs, _, enc_mask = self.model.encoder(feats, lengths)
                return enc_outputs, enc_mask

        encoder_model = ModelEncoderWrapper(self.torch_model)
        encoder_model.eval()
        if not self.ov_encoder_path.exists() :
            example_inputs = {"feats":feats, "lengths":lengths}
            ov_model = convert_model(encoder_model, example_input=example_inputs)
            save_model(ov_model, self.ov_encoder_path, compress_to_fp16=False)
            print(f"✅ ModelEncoder completed {self.ov_encoder_path}")
            del ov_model
            cleanup_torchscript_cache()

        enc_outputs, enc_mask = encoder_model(feats, lengths)

        # if not self.ov_decoder_path.exists() :
        #     class ModelDecoderWrapper(torch.nn.Module):
        #         def __init__(self, model):
        #             super().__init__()
        #             self.model = model.eval()

        #         def forward(self, encoder_outputs, src_mask, ys, B, N, is_finished, scores, softmax_smoothing, caches):
        #             with torch.no_grad():
        #                 t_ys, ys, scores, caches = self.model.decoder.decoder2(encoder_outputs, src_mask,
        #                                                     ys, B, N, is_finished, scores, caches, softmax_smoothing, 1.0)
        #             return t_ys, ys, scores, caches

        #     decoder_model = ModelDecoderWrapper(self.torch_model)
        #     decoder_model.eval()
        #     beam_size=3
        #     num = 2
        #     cache_size = 16
            
        #     B = beam_size
        #     N, Ti, H = enc_outputs.size()
        #     cache_shape = (B*N, num, 1280)

        #     encoder_outputs = enc_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, Ti, H)
        #     src_mask = enc_mask.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, -1, Ti)
        #     ys = torch.ones(N*B, 1).fill_(self.sos_id).long()
        #     scores = torch.tensor([0.0] + [-self.INF]*(B-1)).float()
        #     scores = scores.repeat(N).view(N*B, 1)
        #     is_finished = torch.zeros_like(scores)
 
        #     B = torch.tensor(B).long()
        #     N = torch.tensor(N).long()
        #     softmax_smoothing = torch.tensor(softmax_smoothing).float()

        #     caches: List[Optional[Tensor]] = []
        #     input_names = ["encoder_outputs", "src_mask", "ys", "B", "N", "is_finished", "scores", "softmax_smoothing"]
        #     output_names = ["t_ys", "ys", "scores"]
        #     input_len = len(input_names)
        #     output_len = len(output_names)
        #     for i in range(cache_size):
        #         cache = torch.randn(cache_shape)
        #         caches.append(cache)
        #         input_names.extend([f"key_values.{i}"])
        #         output_names.extend([f"present.{i}"])

        #     example_input = {"encoder_outputs": encoder_outputs, "src_mask": src_mask,  "ys":ys,
        #                      "B": B, "N": N,  "is_finished":is_finished, "scores":scores,
        #                      "softmax_smoothing": softmax_smoothing, "caches": caches}
                
        #     ov_model = ov.convert_model(decoder_model, example_input=example_input)
            
        #     for input, input_name in zip(ov_model.inputs, input_names):
        #         input.get_tensor().set_names({input_name})

        #     for output, output_name in zip(ov_model.outputs, output_names):
        #         output.get_tensor().set_names({output_name})

        #     patch_stateful(ov_model, input_len, output_len)
        #     print("✅ Language model successfully converted")

        #     ov.save_model(ov_model, self.ov_decoder_path, compress_to_fp16=False)
        #     del ov_model
        #     cleanup_torchscript_cache()
        #     print(f"✅ ModelDecoder completed {self.ov_decoder_path}")

        if not self.ov_decoder0_path.exists() :
            class ModelDecoder0Wrapper(torch.nn.Module):
                def __init__(self, model, pad_id, eos_id):
                    super().__init__()
                    self.model = model.eval()
                    self.pad_id = pad_id
                    self.eos_id = eos_id

                def forward(self, ys, encoder_outputs, src_mask, scores, is_finished, softmax_smoothing, eos_penalty, B, N):
                    with torch.no_grad():
                        tgt_mask = self.model.decoder.ignored_target_position_is_0(ys, self.pad_id)
                        dec_output = self.model.decoder.tgt_word_emb(ys) * self.model.decoder.scale + self.model.decoder.positional_encoding(ys)
                        tmp_caches: List[Optional[Tensor]] = []
                        for i, dec_layer in enumerate(self.model.decoder.layer_stack):
                            dec_output = dec_layer.forward0(
                                dec_output, encoder_outputs,
                                tgt_mask, src_mask)
                            tmp_caches.append(dec_output)

                        dec_output = self.model.decoder.layer_norm_out(dec_output)
                        t_logit = self.model.decoder.tgt_word_prj(dec_output[:, -1])
                        # return t_logit, new_caches
                        t_scores = F.log_softmax(t_logit / softmax_smoothing, dim=-1)
                        t_scores[:, self.eos_id] *= eos_penalty
                            
                        t_topB_scores, t_topB_ys = torch.topk(t_scores, k=B, dim=1)
                        t_topB_scores = self.model.decoder.set_finished_beam_score_to_zero(t_topB_scores, is_finished)
                        t_topB_ys = self.model.decoder.set_finished_beam_y_to_eos(t_topB_ys, is_finished)

                        # Accumulated
                        new_scores = scores + t_topB_scores

                        # Pruning
                        new_scores = new_scores.view(N, B*B)
                        new_scores, topB_score_ids = torch.topk(new_scores, k=B, dim=1)
                        new_scores = new_scores.view(-1, 1)

                        topB_row_number_in_each_B_rows_of_ys = torch.div(topB_score_ids, B).view(N*B)
                        stride = B * torch.arange(N).view(N, 1).repeat(1, B).view(N*B)
                        topB_row_number_in_ys = topB_row_number_in_each_B_rows_of_ys.long() + stride.long()

                        # Update ys
                        new_ys = ys[topB_row_number_in_ys]
                        t_ys = torch.gather(t_topB_ys.view(N, B*B), dim=1, index=topB_score_ids).view(N*B, 1)
                        new_ys = torch.cat((new_ys, t_ys), dim=1)

                        # Update caches
                        new_caches: List[Optional[Tensor]] = []
                        for cache in tmp_caches:
                            new_caches.append(cache[topB_row_number_in_ys])
                        return t_ys, new_scores, new_ys, new_caches

            decoder_model = ModelDecoder0Wrapper(self.torch_model, self.pad_id, self.eos_id)
            decoder_model.eval()

            beam_size=3
            num = 2
            cache_size = 16
            
            B = beam_size
            N, Ti, H = enc_outputs.size()
            cache_shape = (B*N, num, 1280)

            encoder_outputs = enc_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, Ti, H)
            src_mask = enc_mask.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, -1, Ti)
            ys = torch.ones(N*B, 1).fill_(self.sos_id).long()
            tgt_mask = self.torch_model.decoder.ignored_target_position_is_0(ys, self.pad_id)
            scores = torch.tensor([0.0] + [-self.INF]*(B-1)).float()
            scores = scores.repeat(N).view(N*B, 1)
            is_finished = torch.zeros_like(scores)
            N = torch.tensor(N).long()
            B = torch.tensor(B).long()

            input_names = ["ys", "encoder_outputs", "src_mask", "scores", "is_finished",
                           "softmax_smoothing", "eos_penalty", "B", "N"]
            output_names = ["t_ys", "new_scores", "new_ys"]
            for i in range(cache_size):
                output_names.extend([f"new_cache.{i}"])

            # ys:torch.Size([3, 1]), tgt_mask:torch.Size([3, 1, 1]), encoder_outputs:torch.Size([3, 673, 1280]), src_mask:torch.Size([3, 1, 673])
            # example_input = {"ys":ys, "tgt_mask":tgt_mask, "encoder_outputs": encoder_outputs, "src_mask": src_mask}
            example_input = {"ys":ys, "encoder_outputs": encoder_outputs, "src_mask": src_mask,  "scores": scores, "is_finished":is_finished,
                             "softmax_smoothing": softmax_smoothing, "eos_penalty": eos_penalty, "B": B, "N": N}
                
            ov_model = ov.convert_model(decoder_model, example_input=example_input)
            
            for input, input_name in zip(ov_model.inputs, input_names):
                input.get_tensor().set_names({input_name})

            for output, output_name in zip(ov_model.outputs, output_names):
                output.get_tensor().set_names({output_name})

            ov.save_model(ov_model, self.ov_decoder0_path, compress_to_fp16=False)
            del ov_model
            cleanup_torchscript_cache()
            print(f"✅ ModelDecoder0 completed {self.ov_decoder0_path}")

        if not self.ov_decoder1_path.exists() :
            class ModelDecoder1Wrapper(torch.nn.Module):
                def __init__(self, model, pad_id, eos_id):
                    super().__init__()
                    self.model = model.eval()
                    self.pad_id = pad_id
                    self.eos_id = eos_id

                def forward(self, ys, encoder_outputs, src_mask, scores, is_finished, softmax_smoothing, eos_penalty, B, N, caches):
                    with torch.no_grad():
                        tgt_mask = self.model.decoder.ignored_target_position_is_0(ys, self.pad_id)
                        dec_output = self.model.decoder.tgt_word_emb(ys) * self.model.decoder.scale + self.model.decoder.positional_encoding(ys)
                        tmp_caches: List[Optional[Tensor]] = []
                        for i, dec_layer in enumerate(self.model.decoder.layer_stack):
                            dec_output = dec_layer.forward1(
                                dec_output, encoder_outputs,
                                tgt_mask, src_mask, caches[i])
                            tmp_caches.append(dec_output)

                        dec_output = self.model.decoder.layer_norm_out(dec_output)
                        t_logit = self.model.decoder.tgt_word_prj(dec_output[:, -1])
                        # return t_logit, new_caches
                        t_scores = F.log_softmax(t_logit / softmax_smoothing, dim=-1)
                        t_scores[:, self.eos_id] *= eos_penalty
                        t_topB_scores, t_topB_ys = torch.topk(t_scores, k=B, dim=1)
                        t_topB_scores = self.model.decoder.set_finished_beam_score_to_zero(t_topB_scores, is_finished)
                        t_topB_ys = self.model.decoder.set_finished_beam_y_to_eos(t_topB_ys, is_finished)

                        # Accumulated
                        new_scores = scores + t_topB_scores

                        # Pruning
                        new_scores = new_scores.view(N, B*B)
                        new_scores, topB_score_ids = torch.topk(new_scores, k=B, dim=1)
                        new_scores = new_scores.view(-1, 1)

                        topB_row_number_in_each_B_rows_of_ys = torch.div(topB_score_ids, B).view(N*B)
                        stride = B * torch.arange(N).view(N, 1).repeat(1, B).view(N*B)
                        topB_row_number_in_ys = topB_row_number_in_each_B_rows_of_ys.long() + stride.long()

                        # Update ys
                        new_ys = ys[topB_row_number_in_ys]
                        t_ys = torch.gather(t_topB_ys.view(N, B*B), dim=1, index=topB_score_ids).view(N*B, 1)
                        new_ys = torch.cat((new_ys, t_ys), dim=1)

                        # Update caches
                        new_caches: List[Optional[Tensor]] = []
                        for cache in tmp_caches:
                            new_caches.append(cache[topB_row_number_in_ys])
                        return t_ys, new_scores, new_ys, new_caches

            decoder_model = ModelDecoder1Wrapper(self.torch_model, self.pad_id, self.eos_id)
            decoder_model.eval()

            beam_size=3
            num = 2
            cache_size = 16
            
            B = beam_size
            N, Ti, H = enc_outputs.size()
            cache_shape = (B*N, num, 1280)

            encoder_outputs = enc_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, Ti, H)
            src_mask = enc_mask.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, -1, Ti)
            ys = torch.ones(N*B, num+1).fill_(self.sos_id).long()
            tgt_mask = self.torch_model.decoder.ignored_target_position_is_0(ys, self.pad_id)
            scores = torch.tensor([0.0] + [-self.INF]*(B-1)).float()
            scores = scores.repeat(N).view(N*B, 1)
            is_finished = torch.zeros_like(scores)
            N = torch.tensor(N).long()
            B = torch.tensor(B).long()

            caches = []

            input_names = ["ys", "encoder_outputs", "src_mask", "scores", "is_finished",
                           "softmax_smoothing", "eos_penalty", "B", "N"]
            output_names = ["t_ys","new_scores", "new_ys"]

            for i in range(cache_size):
                cache = torch.randn(cache_shape)
                caches.append(cache)
                input_names.extend([f"cache.{i}"])
                output_names.extend([f"new_cache.{i}"])

            # ys:torch.Size([3, 1]), tgt_mask:torch.Size([3, 1, 1]), encoder_outputs:torch.Size([3, 673, 1280]), src_mask:torch.Size([3, 1, 673])
            example_input = {"ys":ys, "encoder_outputs": encoder_outputs, "src_mask": src_mask,  "scores": scores, "is_finished":is_finished,
                             "softmax_smoothing": softmax_smoothing, "eos_penalty": eos_penalty, "B": B, "N": N,
                             "caches": caches}
                
            ov_model = ov.convert_model(decoder_model, example_input=example_input)
            
            for input, input_name in zip(ov_model.inputs, input_names):
                input.get_tensor().set_names({input_name})

            for output, output_name in zip(ov_model.outputs, output_names):
                output.get_tensor().set_names({output_name})

            ov.save_model(ov_model, self.ov_decoder1_path, compress_to_fp16=False)
            del ov_model
            cleanup_torchscript_cache()
            print(f"✅ ModelDecoder1 completed {self.ov_decoder1_path}")
            
        with open(self.ov_config_path, "w") as file:
            data = {
                "sos_id": self.sos_id,
                "eos_id": self.eos_id,
                "pad_id": self.pad_id,
            }
            json.dump(data, file, indent=2)
            print(f"✅ Save model config to {self.ov_config_path}")