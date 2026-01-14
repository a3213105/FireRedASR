import torch
import time
from fireredasr.models.module.conformer_encoder import ConformerEncoder
from fireredasr.models.module.transformer_decoder import TransformerDecoder
import json
import os

# from memory_profiler import profile

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
        # t0 = time.perf_counter()
        enc_outputs, _, enc_mask = self.encoder(padded_input, input_lengths)
        # t1 = time.perf_counter()
        nbest_hyps = self.decoder.batch_beam_search(
            enc_outputs, enc_mask,
            beam_size, nbest, decode_max_len,
            softmax_smoothing, length_penalty, eos_penalty)
        # t2 = time.perf_counter()
        # print(f"### enc:{t1-t0:.5f}, dec:{t2-t1:.5f}, "
        #       f"{nbest_hyps[0][0]['yseq'].shape}")
        return nbest_hyps
    
    def encode(self, padded_input, input_lengths):
        enc_outputs, _, enc_mask = self.encoder(padded_input, input_lengths)
        return enc_outputs, enc_mask

class FireRedAsrAed1(torch.nn.Module):
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
        # t0 = time.perf_counter()
        enc_outputs, _, enc_mask = self.encoder(padded_input, input_lengths)
        # t1 = time.perf_counter()
        nbest_hyps = self.decoder.batch_beam_search1(
            enc_outputs, enc_mask,
            beam_size, nbest, decode_max_len,
            softmax_smoothing, length_penalty, eos_penalty)
        # t2 = time.perf_counter()
        # print(f"### enc:{t1-t0:.5f}, dec:{t2-t1:.5f}, "
        #       f"{nbest_hyps[0][0]['yseq'].shape}")
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

class base_torch_function_ov :
    def eval(self):
        return self
    
    def cpu(self):
        return self

FireRedAsrAed_CONFIG_NAME = "FireRedASR_AED_config.json"
FireRedAsrAed_MODEL_NAME = "FireRedASR_AED_ov.xml"
FireRedAsrAed_Encoder_MODEL_NAME = "FireRedASR_AED_encoder_ov.xml"
FireRedAsrAed_Decoder_MODEL_NAME = "FireRedASR_AED_decoder_ov.xml"
FireRedAsrAed_Decoder0_MODEL_NAME = "FireRedASR_AED_decoder0_ov.xml"
FireRedAsrAed_Decoder1_MODEL_NAME = "FireRedASR_AED_decoder1_ov.xml"
           
class FireRedAsrAed_ov(base_torch_function_ov):
    def __init__(self, args, ov_core, model_path, enc_type, dec_type, cache_size):
        self.ov_version = "ov_model_v0"
        self.init(args, ov_core, model_path, enc_type, dec_type, cache_size)
        
    def init(self, args, ov_core, model_path, enc_type, dec_type, cache_size):
        ov_path = Path(model_path)
        self.init_params(args, ov_core, ov_path)
        self.init_model_path(ov_path)
        self.enc_type = enc_type
        self.dec_type = dec_type
        self.cache_size = cache_size
        if self.enc_type in "f32f16bf16" and self.dec_type in "f32f16bf16" :
            self.load_ov_model()

    def init_model_path(self, ov_path):
        self.ov_encoder_path = ov_path.parent / self.ov_version / FireRedAsrAed_Encoder_MODEL_NAME
        self.ov_decoder0_path = ov_path.parent / self.ov_version / FireRedAsrAed_Decoder0_MODEL_NAME
        self.ov_decoder1_path = ov_path.parent / self.ov_version / FireRedAsrAed_Decoder1_MODEL_NAME
        if not self.ov_encoder_path.exists() or not self.ov_decoder0_path.exists() or not self.ov_decoder1_path.exists():
            self.converted_to_ov = True

    def init_params(self, args, ov_core, ov_path) :
        self.ov_config_path = ov_path.parent / self.ov_version / FireRedAsrAed_CONFIG_NAME
        self.sos_id = 3
        self.eos_id = 4
        self.pad_id = 2
        if args is None :
            self.load_config()
        else :
            self.sos_id = args.sos_id
            self.eos_id = args.eos_id
            self.pad_id = args.pad_id
        self.INF = 1e10
        self.next_beam_idx = None
        self.torch_model = None
        self.converted_to_ov = False
        self.using_ov = False
        self.ov_core = ov_core

    def load_ov_model(self):
        try :
            if self.ov_core is None :
                self.ov_core = ov.Core()
            cache_size_str = f"{self.cache_size}"
            self.ov_core.set_property("CPU", {"CPU_RUNTIME_CACHE_CAPACITY": cache_size_str})
            ov_config = {'INFERENCE_PRECISION_HINT': self.enc_type,'PERFORMANCE_HINT': 'LATENCY',}
            self.ov_encoder_model = self.ov_core.compile_model(self.ov_encoder_path, 'CPU', ov_config)
            ov_config = {'INFERENCE_PRECISION_HINT': self.dec_type, 'PERFORMANCE_HINT': "LATENCY"}
            self.ov_decoder0_model = self.ov_core.compile_model(self.ov_decoder0_path, 'CPU', ov_config)
            self.ov_decoder1_model = self.ov_core.compile_model(self.ov_decoder1_path, 'CPU', ov_config)
            
            self.enc_request = self.ov_encoder_model.create_infer_request()
            self.dec0_request = self.ov_decoder0_model.create_infer_request()
            self.dec1_request = self.ov_decoder1_model.create_infer_request()
            self.using_ov = True
        except Exception as e:
            print(f"### ov load {self.ov_encoder_path} or {self.ov_decoder0_path} or {self.ov_decoder1_path} failed, {e}")

    def get_ys_lengths(self, ys):
        # N, B, Tmax = ys.size()
        ys_lengths = torch.sum(torch.ne(ys, self.eos_id), dim=-1)
        return ys_lengths.int()
    
    def batch_beam_search_for0_ov(self, ys, encoder_outputs, src_mask, scores, is_finished,
                                  softmax_smoothing, eos_penalty, B, N) :
        inputs = (ys, encoder_outputs, src_mask, scores, is_finished, softmax_smoothing, eos_penalty, B, N)
        self.dec0_request.start_async(inputs, share_inputs=True)
        self.dec0_request.wait()
        t_ys = self.dec0_request.get_output_tensor(0).data
        scores = self.dec0_request.get_output_tensor(1)
        ys = self.dec0_request.get_output_tensor(2)
        new_caches = []
        for i in range(3, 19):
            new_caches.append(self.dec0_request.get_output_tensor(i))
        return t_ys, scores, ys, new_caches

        # res = self.ov_decoder0_model((ys, encoder_outputs, src_mask, scores, is_finished, softmax_smoothing, eos_penalty, B, N), share_inputs = True, share_outputs = True)
        # new_caches = []
        # for i in range(3, len(res)):
        #     new_caches.append(res[i])
        # return res[0], res[1], res[2], new_caches

    def batch_beam_search_for1_ov(self, ys, encoder_outputs, src_mask, scores, is_finished,
                                  softmax_smoothing, eos_penalty, B, N, caches) :
        inputs = (ys, encoder_outputs, src_mask, scores, is_finished, softmax_smoothing, eos_penalty, B, N, *caches)
        self.dec1_request.start_async(inputs, share_inputs=True)
        self.dec1_request.wait()
        t_ys = self.dec1_request.get_output_tensor(0).data
        scores = self.dec1_request.get_output_tensor(1)
        ys = self.dec1_request.get_output_tensor(2)
        new_caches = []
        for i in range(3, 19):
            new_caches.append(self.dec1_request.get_output_tensor(i))
        return t_ys, scores, ys, new_caches

        # res = self.ov_decoder1_model((ys, encoder_outputs, src_mask, scores, is_finished, softmax_smoothing, eos_penalty, B, N, *caches), share_inputs = True, share_outputs = True)
        # new_caches = []
        # for i in range(3, len(res)):
        #     new_caches.append(res[i])
        # return res[0], res[1], res[2], new_caches

    # @profile
    def batch_beam_search_ov(self, encoder_outputs, src_masks,
                   beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):
        B = beam_size
        N, Ti, H = encoder_outputs.shape
        # device = encoder_outputs.device
        maxlen = decode_max_len if decode_max_len > 0 else Ti
        assert eos_penalty > 0.0 and eos_penalty <= 1.0

        # Init
        encoder_outputs = np.repeat(np.expand_dims(encoder_outputs, axis=1), repeats=B, axis=1).reshape(N * B, Ti, H)
        src_mask = np.repeat(np.expand_dims(src_masks, axis=1), repeats=B, axis=1).reshape(N * B, -1, Ti)   
        ys = np.full((N * B, 1), fill_value=self.sos_id, dtype=np.int64)
        t_ys = ys
        scores = np.tile(np.array([0.0] + [-self.INF] * (B - 1), dtype=np.float32), reps=N).reshape(N * B, 1)
        is_finished = np.zeros_like(scores, dtype=np.float32)

        # encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, Ti, H)
        # src_mask = src_masks.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, -1, Ti)
        # ys = torch.ones(N*B, 1).fill_(self.sos_id).long()
        # scores = torch.tensor([0.0] + [-self.INF]*(B-1)).float()
        # scores = scores.repeat(N).view(N*B, 1)
        # is_finished = torch.zeros_like(scores)

        t_ys, scores, ys, caches = self.batch_beam_search_for0_ov(ys, encoder_outputs, src_mask,
                                                                  scores, is_finished, softmax_smoothing,
                                                                  eos_penalty, B, N)
        t_ys = torch.from_numpy(t_ys)
        # Update finished state
        is_finished = t_ys.eq(self.eos_id)
        if is_finished.sum().item() != N*B:
            # Autoregressive Prediction
            for t in range(1, maxlen):
                t_ys, scores, ys, caches = self.batch_beam_search_for1_ov(ys, encoder_outputs, src_mask,
                                                                          scores, is_finished,
                                                                          softmax_smoothing, eos_penalty,
                                                                          B, N, caches)
                t_ys = torch.from_numpy(t_ys)
                # Update finished state
                is_finished = t_ys.eq(self.eos_id)
                if is_finished.sum().item() == N*B:
                    break
        scores = torch.from_numpy(scores.data)
        ys = torch.from_numpy(ys.data)

        scores = scores.view(N, B)
        ys = ys.view(N, B, -1)
        ys_lengths = self.get_ys_lengths(ys)
        if length_penalty > 0.0:
            penalty = torch.pow((5+ys_lengths.float())/(5.0+1), length_penalty)
            scores /= penalty
        nbest_scores, nbest_ids = torch.topk(scores, k=int(nbest), dim=1)
        nbest_scores = -1.0 * nbest_scores
        index = nbest_ids + B * torch.arange(N).view(N, 1).long()
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

    def transcribe_ov(self, padded_input, input_lengths,
                   beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):
        # if self.converted_to_ov :
        #     self.convert_ov_model(padded_input, input_lengths, beam_size, nbest, decode_max_len,
        #         softmax_smoothing, length_penalty, eos_penalty)
        #     self.load_ov_model()

        # print(f"inputs shape: padded_input:{padded_input.shape}, input_lengths:{input_lengths.shape}")
        # t0 = time.perf_counter()
        inputs = (padded_input, input_lengths)
        self.enc_request.start_async(inputs, share_inputs=True)
        self.enc_request.wait()
        # t1 = time.perf_counter()
        # enc_outputs = torch.from_numpy(self.enc_request.get_output_tensor(0).data)
        # enc_mask = torch.from_numpy(self.enc_request.get_output_tensor(1).data)
        enc_outputs = self.enc_request.get_output_tensor(0).data
        enc_mask = self.enc_request.get_output_tensor(1).data
        nbest_hyps = self.batch_beam_search_ov(enc_outputs, enc_mask, beam_size, nbest,decode_max_len, softmax_smoothing, length_penalty, eos_penalty)
        # t2 = time.perf_counter()
        # steps= nbest_hyps[0][0]['yseq'].shape[0]
        # enc_t = t1-t0
        # dec_t = t2-t1
        # print(f"### model input:{padded_input.shape[1]}, "
        #       f"enc:{enc_t:.4f}, dec:{dec_t:.4f}, {dec_t/steps:.4f}, {steps}")
        return nbest_hyps

    def transcribe(self, padded_input, input_lengths,
                   beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):
        if self.converted_to_ov :
            self.convert_ov_model(padded_input, input_lengths, beam_size, nbest, decode_max_len,
                softmax_smoothing, length_penalty, eos_penalty)
            self.load_ov_model()
        if self.using_ov :
            return self.transcribe_ov(padded_input, input_lengths,
                   beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty)
        else :
            return self.torch_model.transcribe(padded_input, input_lengths,
                   beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty)

    def load_config(self):
        # print(f"Load model config from {self.ov_config_path}")
        try :
            with open(self.ov_config_path, 'r') as file:
                data = json.load(file)
                self.sos_id = data["sos_id"]
                self.eos_id = data["eos_id"]
                self.pad_id = data["pad_id"]
                return True
        except :
            print(f"{self.ov_config_path} is not existed")
        finally:
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

        if not self.ov_decoder0_path.exists() :
            class ModelDecoder0Wrapper(torch.nn.Module):
                def __init__(self, model, pad_id, eos_id):
                    super().__init__()
                    self.model = model.eval()
                    self.pad_id = pad_id
                    self.eos_id = eos_id

                def forward(self, ys, encoder_outputs, src_mask, scores, is_finished, softmax_smoothing, eos_penalty, B, N):
                    with torch.no_grad():
                        tgt_mask = self.model.decoder.ignored_target_position_is(ys, self.pad_id)
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
            tgt_mask = self.torch_model.decoder.ignored_target_position_is(ys, self.pad_id)
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
                        tgt_mask = self.model.decoder.ignored_target_position_is(ys, self.pad_id)
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
            tgt_mask = self.torch_model.decoder.ignored_target_position_is(ys, self.pad_id)
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
            
        if not os.path.exists(self.ov_config_path):
            with open(self.ov_config_path, "w") as file:
                data = {
                    "sos_id": self.sos_id,
                    "eos_id": self.eos_id,
                    "pad_id": self.pad_id,
                }
                json.dump(data, file, indent=2)
                print(f"✅ Save model config to {self.ov_config_path}")

# class FireRedAsrAed_ov1(FireRedAsrAed_ov) :
#     def __init__(self, args, ov_core, model_path, enc_type, dec_type, cache_size):
#         self.ov_version = "ov_model_v1"
#         self.init(args, ov_core, model_path, enc_type, dec_type, cache_size)

#     def init_model_path(self, ov_path):
#         self.ov_encoder_path = ov_path.parent / self.ov_version / FireRedAsrAed_Encoder_MODEL_NAME
#         self.ov_decoder_path = ov_path.parent / self.ov_version / FireRedAsrAed_Decoder_MODEL_NAME
#         if not self.ov_encoder_path.exists() or not self.ov_decoder_path.exists():
#             self.converted_to_ov = True
        
#     def load_ov_model(self):
#         try :
#             if self.ov_core is None :
#                 self.ov_core = ov.Core()
#             cache_size_str = f"{self.cache_size}"
#             self.ov_core.set_property("CPU", {"CPU_RUNTIME_CACHE_CAPACITY": cache_size_str})
#             ov_config = {'INFERENCE_PRECISION_HINT': self.enc_type,'PERFORMANCE_HINT': 'LATENCY',}
#             self.ov_encoder_model = self.ov_core.compile_model(self.ov_encoder_path, 'CPU', ov_config)
#             ov_config = {'INFERENCE_PRECISION_HINT': self.dec_type, 'PERFORMANCE_HINT': "LATENCY"}
#             self.ov_decoder_model = self.ov_core.compile_model(self.ov_decoder_path, 'CPU', ov_config)
            
#             self.enc_request = self.ov_encoder_model.create_infer_request()
#             self.dec_request = self.ov_decoder_model.create_infer_request()
#             self.using_ov = True
#         except Exception as e:
#             print(f"### ov load {self.ov_encoder_path} or {self.ov_decoder_path} failed, {e}")

#     def transcribe_ov(self, padded_input, input_lengths,
#                    beam_size, nbest, decode_max_len,
#                    softmax_smoothing, length_penalty, eos_penalty):
#         # print(f"inputs shape: padded_input:{padded_input.shape}, input_lengths:{input_lengths.shape}")
#         # t0 = time.perf_counter()
#         inputs = (padded_input, input_lengths)
#         self.enc_request.start_async(inputs, share_inputs=True)
#         self.dec_request.reset_state()
#         self.enc_request.wait()
#         # t1 = time.perf_counter()

#         encoder_outputs = self.enc_request.get_output_tensor(0).data
#         src_masks = self.enc_request.get_output_tensor(1).data

#         B = beam_size
#         N, Ti, H = encoder_outputs.shape
#         maxlen = decode_max_len if decode_max_len > 0 else Ti

#         # Init       
#         encoder_outputs = np.repeat(np.expand_dims(encoder_outputs, axis=1), repeats=B, axis=1).reshape(N * B, Ti, H)
#         src_mask = np.repeat(np.expand_dims(src_masks, axis=1), repeats=B, axis=1).reshape(N * B, -1, Ti)   
#         ys = np.full((N * B, 1), fill_value=self.sos_id, dtype=np.int64)
#         t_ys = ys
#         scores = np.tile(np.array([0.0] + [-self.INF] * (B - 1), dtype=np.float32), reps=N).reshape(N * B, 1)
#         is_finished = np.zeros_like(scores, dtype=np.float32)

#         self.next_beam_idx = np.arange(B, dtype=int)
#         for t in range(maxlen):
#             self.dec_request.start_async({"t_ys" : t_ys, "encoder_outputs" : encoder_outputs, "src_mask" : src_mask,
#                     "softmax_smoothing": softmax_smoothing, "eos_penalty": eos_penalty,
#                     "is_finished": is_finished, "B" : B, "N" : N, "scores" : scores,
#                     "beam_idx" : self.next_beam_idx}, share_inputs=True)
#             self.dec_request.wait()
#             topB_row_number_in_ys = self.dec_request.get_tensor("topB_row_number_in_ys").data
#             t_ys = self.dec_request.get_tensor("new_t_ys").data
#             scores = self.dec_request.get_tensor("new_scores").data
#             ys = ys[topB_row_number_in_ys]
#             ys = np.concatenate((ys, t_ys), axis=1)
#             is_finished = (t_ys == self.eos_id) 
#             if int(is_finished.sum()) == N * B:
#                 break
#         # Length penalty (follow GNMT)
#         scores = torch.from_numpy(scores)
#         ys = torch.from_numpy(ys)
#         scores = scores.view(N, B)
#         ys = ys.view(N, B, -1)
#         ys_lengths = self.get_ys_lengths(ys)
#         if length_penalty > 0.0:
#             penalty = torch.pow((5+ys_lengths.float())/(5.0+1), length_penalty)
#             scores /= penalty
#         nbest_scores, nbest_ids = torch.topk(scores, k=int(nbest), dim=1)
#         nbest_scores = -1.0 * nbest_scores
#         index = nbest_ids + B * torch.arange(N).view(N, 1).long()
#         nbest_ys = ys.view(N*B, -1)[index.view(-1)]
#         nbest_ys = nbest_ys.view(N, nbest_ids.size(1), -1)
#         nbest_ys_lengths = ys_lengths.view(N*B)[index.view(-1)].view(N, -1)

#         # result
#         nbest_hyps: List[List[Dict[str, Tensor]]] = []
#         for n in range(N):
#             n_nbest_hyps: List[Dict[str, Tensor]] = []
#             for i, score in enumerate(nbest_scores[n]):
#                 new_hyp = {
#                     "yseq": nbest_ys[n, i, 1:nbest_ys_lengths[n, i]]
#                 }
#                 n_nbest_hyps.append(new_hyp)
#             nbest_hyps.append(n_nbest_hyps)
#         return nbest_hyps
#         # t2 = time.perf_counter()
#         # steps= nbest_hyps[0][0]['yseq'].shape[0]
#         # enc_t = t1-t0
#         # dec_t = t2-t1
#         # print(f"### model input:{padded_input.shape[1]}, "
#         #       f"enc:{enc_t:.4f}, dec:{dec_t:.4f}, {dec_t/steps:.4f}, {steps}")
#         # return nbest_hyps

class FireRedAsrAed_ov2(FireRedAsrAed_ov) :
    def __init__(self, args, ov_core, model_path, enc_type, dec_type, cache_size):
        self.ov_version = "ov_model_v2"
        self.init(args, ov_core, model_path, enc_type, dec_type, cache_size)

    def init_model_path(self, ov_path):
        self.ov_encoder_path = ov_path.parent / self.ov_version / FireRedAsrAed_Encoder_MODEL_NAME
        self.ov_decoder_path = ov_path.parent / self.ov_version / FireRedAsrAed_Decoder_MODEL_NAME
        if not self.ov_encoder_path.exists() or not self.ov_decoder_path.exists():
            self.converted_to_ov = True
        
    def load_ov_model(self):
        try :
            if self.ov_core is None :
                self.ov_core = ov.Core()
            cache_size_str = f"{self.cache_size}"
            self.ov_core.set_property("CPU", {"CPU_RUNTIME_CACHE_CAPACITY": cache_size_str})
            ov_config = {'INFERENCE_PRECISION_HINT': self.enc_type,'PERFORMANCE_HINT': 'LATENCY',}
            self.ov_encoder_model = self.ov_core.compile_model(self.ov_encoder_path, 'CPU', ov_config)
            ov_config = {'INFERENCE_PRECISION_HINT': self.dec_type, 'PERFORMANCE_HINT': "LATENCY"}
            self.ov_decoder_model = self.ov_core.compile_model(self.ov_decoder_path, 'CPU', ov_config)
            
            self.enc_request = self.ov_encoder_model.create_infer_request()
            self.dec_request = self.ov_decoder_model.create_infer_request()
            self.using_ov = True
        except Exception as e:
            print(f"### ov load {self.ov_encoder_path} or {self.ov_decoder_path} failed, {e}")

    def transcribe_ov(self, padded_input, input_lengths,
                   beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):
        # t0 = time.perf_counter()
        inputs = (padded_input, input_lengths)
        self.enc_request.start_async(inputs, share_inputs=True)
        self.dec_request.reset_state()
        self.enc_request.wait()
        # t1 = time.perf_counter()

        encoder_outputs = self.enc_request.get_output_tensor(0).data
        src_masks = self.enc_request.get_output_tensor(1).data

        B = beam_size
        N, Ti, H = encoder_outputs.shape
        maxlen = decode_max_len if decode_max_len > 0 else Ti

        # Init       
        encoder_outputs = np.repeat(np.expand_dims(encoder_outputs, axis=1), repeats=B, axis=1).reshape(N * B, Ti, H)
        src_mask = np.repeat(np.expand_dims(src_masks, axis=1), repeats=B, axis=1).reshape(N * B, -1, Ti)   
        ys = np.full((N * B, 1), fill_value=self.sos_id, dtype=np.int64)
        t_ys = ys
        scores = np.tile(np.array([0.0] + [-self.INF] * (B - 1), dtype=np.float32), reps=N).reshape(N * B, 1)
        is_finished = np.zeros_like(scores, dtype=np.float32)
        scores_mask = np.tile(np.array([0.0] + [-self.INF] * (B - 1), dtype=np.float32), reps=N).reshape(N * B, 1)
        self.next_beam_idx = np.arange(B, dtype=int)
        for t in range(maxlen):
            self.dec_request.start_async({"t_ys" : t_ys, "encoder_outputs" : encoder_outputs,
                                          "src_mask" : src_mask, "softmax_smoothing": softmax_smoothing,
                                          "eos_penalty": eos_penalty, "is_finished": is_finished,
                                          "B" : B, "N" : N, "scores" : scores, "scores_mask" : scores_mask,
                                          "beam_idx" : self.next_beam_idx},
                                         share_inputs=True)
            self.dec_request.wait()
            topB_row_number_in_ys = self.dec_request.get_tensor("topB_row_number_in_ys").data
            t_ys = self.dec_request.get_tensor("new_t_ys").data
            scores = self.dec_request.get_tensor("new_scores").data
            ys = ys[topB_row_number_in_ys]
            ys = np.concatenate((ys, t_ys), axis=1)
            is_finished = (t_ys == self.eos_id) 
            if int(is_finished.sum()) == N * B:
                break

        # Length penalty (follow GNMT)
        scores = torch.from_numpy(scores)
        ys = torch.from_numpy(ys)
        scores = scores.view(N, B)
        ys = ys.view(N, B, -1)
        ys_lengths = self.get_ys_lengths(ys)
        if length_penalty > 0.0:
            penalty = torch.pow((5+ys_lengths.float())/(5.0+1), length_penalty)
            scores /= penalty
        nbest_scores, nbest_ids = torch.topk(scores, k=int(nbest), dim=1)
        nbest_scores = -1.0 * nbest_scores
        index = nbest_ids + B * torch.arange(N).view(N, 1).long()
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
        # t2 = time.perf_counter()
        # steps= nbest_hyps[0][0]['yseq'].shape[0]
        # enc_t = t1-t0
        # dec_t = t2-t1
        # print(f"### model input:{padded_input.shape[1]}, "
        #       f"enc:{enc_t:.4f}, dec:{dec_t:.4f}, {dec_t/steps:.4f}, {steps}")
        return nbest_hyps

    def convert_ov_model(self, feats, lengths, beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):
        
        if not os.path.exists(self.ov_config_path):
            folder_path = os.path.dirname(self.ov_config_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            with open(self.ov_config_path, "w") as file:
                data = {
                    "sos_id": self.sos_id,
                    "eos_id": self.eos_id,
                    "pad_id": self.pad_id,
                }
                json.dump(data, file, indent=2)
                print(f"✅ Save model config to {self.ov_config_path}")

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

        if not self.ov_decoder_path.exists() :
            class ModelDecoderWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model.eval()

                def forward(self, t_ys, encoder_outputs, src_mask, softmax_smoothing, eos_penalty,
                            is_finished, B, N, scores, scores_mask, caches):
                    with torch.no_grad():
                        topB_row_number_in_ys, t_ys, scores, caches = self.model.decoder.infer_decoder_mask(t_ys, 
                                            encoder_outputs, src_mask, scores_mask, caches, scores,
                                            softmax_smoothing, eos_penalty, is_finished, B, N)
                        return topB_row_number_in_ys, t_ys, scores, caches

            decoder_model = ModelDecoderWrapper(self.torch_model)
            decoder_model.eval()

            beam_size=3
            num = 2
            cache_size = 16
            
            B = beam_size
            N, Ti, H = enc_outputs.size()
            cache_shape = (B*N, num, 1280)

            encoder_outputs = enc_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, Ti, H)
            src_mask = enc_mask.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, -1, Ti)
            t_ys = torch.ones(N*B, 1).fill_(self.sos_id).long()
            scores = torch.tensor([0.0] + [-self.INF]*(B-1)).float()
            scores = scores.repeat(N).view(N*B, 1)
            # scores = np.tile(np.array([0.0] + [-self.INF] * (B - 1), dtype=np.float32), reps=N).reshape(N * B, 1)
            scores_mask = np.tile(np.array([0.0] + [-self.INF] * (B - 1), dtype=np.float32), reps=N).reshape(N * B, 1)
            is_finished = torch.zeros_like(scores)
            N = torch.tensor(N).long()
            B = torch.tensor(B).long()

            caches = []
            input_names = ["t_ys", "encoder_outputs", "src_mask", "softmax_smoothing",
                           "eos_penalty", "is_finished", "B", "N", "scores", "scores_mask"]
            output_names = ["topB_row_number_in_ys", "new_t_ys", "new_scores"]

            for i in range(cache_size):
                cache = torch.randn(cache_shape)
                caches.append(cache)
                input_names.extend([f"key_values.{i}"])
                output_names.extend([f"present.{i}"])

            example_input = {"t_ys":t_ys, "encoder_outputs": encoder_outputs, "src_mask": src_mask,
                             "softmax_smoothing": softmax_smoothing, "eos_penalty": eos_penalty,
                             "is_finished":is_finished, "B": B, "N": N, "scores": scores,
                             "scores_mask": scores_mask, "caches": caches}
                
            ov_model = ov.convert_model(decoder_model, example_input=example_input)
            
            for input, input_name in zip(ov_model.inputs, input_names):
                input.get_tensor().set_names({input_name})

            for output, output_name in zip(ov_model.outputs, output_names):
                output.get_tensor().set_names({output_name})

            patch_stateful(ov_model, 10, 3)
            print("✅ ModelDecoder model successfully converted")

            ov_model.set_rt_info("f16", ["runtime_options", "KV_CACHE_PRECISION"])
            ov.save_model(ov_model, self.ov_decoder_path, compress_to_fp16=False)
            del ov_model
            cleanup_torchscript_cache()
            print(f"✅ ModelDecoder completed {self.ov_decoder_path}")


from .ov_operator_async import FireRedAsrAedEncDecModel
from .ov_model_helper import FireRedAsrAedConverterWrapper
class FireRedAsrAed_ov1(FireRedAsrAed_ov):
    def __init__(self, args, ov_core, model_path, enc_type, dec_type, cache_size):
        self.ov_model = FireRedAsrAedEncDecModel(args, ov_core, model_path, enc_type, dec_type, cache_size, "ov_model_v1")
        self.init_params(args)
        self.using_ov = self.ov_model.using_ov

    def init_params(self, args) :
        self.INF = 1e10
        if args is not None :
            self.sos_id = args.sos_id
            self.eos_id = args.eos_id
            self.pad_id = args.pad_id
        else :
            self.sos_id = 3
            self.eos_id = 4
            self.pad_id = 2

    def transcribe(self, padded_input, input_lengths,
                   beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):
        if self.ov_model.converted_to_ov :
            converter = FireRedAsrAedConverterWrapper(self.torch_model)
            converter.convert_ov_model(padded_input, input_lengths, beam_size, nbest, decode_max_len,
                softmax_smoothing, length_penalty, eos_penalty,
                self.ov_model.ov_encoder_path, self.ov_model.ov_decoder_path,
                self.sos_id, self.eos_id, self.INF, self.pad_id)
            self.ov_model.load_ov_model()

        B = beam_size

        inputs = (padded_input, input_lengths)
        encoder_outputs, src_masks = self.ov_model.encoder(inputs, B)

        N, Ti, H = encoder_outputs.shape
        maxlen = decode_max_len if decode_max_len > 0 else Ti

        # Init       
        encoder_outputs = np.repeat(np.expand_dims(encoder_outputs, axis=1), repeats=B, axis=1).reshape(N * B, Ti, H)
        src_mask = np.repeat(np.expand_dims(src_masks, axis=1), repeats=B, axis=1).reshape(N * B, -1, Ti)   
        ys = np.full((N * B, 1), fill_value=self.sos_id, dtype=np.int64)
        t_ys = ys
        scores = np.tile(np.array([0.0] + [-self.INF] * (B - 1), dtype=np.float32), reps=N).reshape(N * B, 1)
        is_finished = np.zeros_like(scores, dtype=np.float32)
        # scores_mask = scores
        for t in range(maxlen):
            topB_row_number_in_ys, t_ys, scores = self.ov_model.decoder({"t_ys" : t_ys, "encoder_outputs" : encoder_outputs,
                              "src_mask" : src_mask, "softmax_smoothing": softmax_smoothing,
                              "eos_penalty": eos_penalty, "is_finished": is_finished,
                              "B" : B, "N" : N, "scores" : scores}) #, "scores_mask" : scores_mask
            ys = ys[topB_row_number_in_ys]
            ys = np.concatenate((ys, t_ys), axis=1)
            is_finished = (t_ys == self.eos_id) 
            if int(is_finished.sum()) == N * B:
                break

        # Length penalty (follow GNMT)
        scores = torch.from_numpy(scores)
        ys = torch.from_numpy(ys)
        scores = scores.view(N, B)
        ys = ys.view(N, B, -1)
        ys_lengths = self.get_ys_lengths(ys)
        if length_penalty > 0.0:
            penalty = torch.pow((5+ys_lengths.float())/(5.0+1), length_penalty)
            scores /= penalty
        nbest_scores, nbest_ids = torch.topk(scores, k=int(nbest), dim=1)
        nbest_scores = -1.0 * nbest_scores
        index = nbest_ids + B * torch.arange(N).view(N, 1).long()
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
