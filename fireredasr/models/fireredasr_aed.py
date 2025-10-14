import torch

from fireredasr.models.module.conformer_encoder import ConformerEncoder
from fireredasr.models.module.transformer_decoder import TransformerDecoder


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

from pathlib import Path
import openvino as ov
FireRedAsrAed_MODEL_NAME = "FireRedASR_AED_ov.xml"
class FireRedAsrAed_ov :
    def __init__(self, ov_core, model_path, infer_type):
        self.torch_model = None
        self.converted_to_ov = False
        try :
            if ov_core is None :
                ov_core = ov.Core()
            self.ov_path = Path(model_path)
            self.ov_path = self.ov_path.parent / "ov_model" / FireRedAsrAed_MODEL_NAME
            if not self.ov_path.exists() :
                self.converted_to_ov = True
            self.ov_dtype = infer_type
            ov_config = {'INFERENCE_PRECISION_HINT':self.ov_dtype, 'PERFORMANCE_HINT': "LATENCY"}
            self.ov_model = ov_core.compile_model(self.ov_path, 'CPU', ov_config)
            self.using_ov = True
        except Exception as e:
            self.using_ov = False
            self.ov_model = None
            print(f"### ov load {self.ov_path} failed, {e}")


    @torch.inference_mode()
    def transcribe(self, padded_input, input_lengths,
                   beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):
        beam_size = torch.tensor(beam_size, dtype=torch.int32)
        nbest = torch.tensor(nbest, dtype=torch.int32)
        decode_max_len = torch.tensor(decode_max_len, dtype=torch.int32)
        softmax_smoothing = torch.tensor(softmax_smoothing, dtype=torch.float32)
        length_penalty = torch.tensor(length_penalty, dtype=torch.float32)
        eos_penalty = torch.tensor(eos_penalty, dtype=torch.float32)

        if self.using_ov :
            # inputs = (padded_input, input_lengths, beam_size, nbest, decode_max_len,
            #         softmax_smoothing, length_penalty, eos_penalty)
            inputs = (padded_input, input_lengths)
            res = self.ov_model(inputs)
            nbest_scores = res[0]
            nbest_ys = res[1]
            nbest_ys_lengths = res[2]
            N = padded_input.size(0)
            nbest_hyps = self.batch_beam_search_2_list(N, nbest_scores, nbest_ys, nbest_ys_lengths)
        else :
            nbest_hyps = self.torch_model.transcribe(padded_input, input_lengths,
                   beam_size, nbest, decode_max_len, softmax_smoothing, length_penalty, eos_penalty)
            if self.converted_to_ov :
                self.convert_ov_model(padded_input, input_lengths, beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty)
        return nbest_hyps


    def batch_beam_search_2_list(self, N, nbest_scores, nbest_ys, nbest_ys_lengths):
        nbest_hyps: List[List[Dict[str, Tensor]]] = []
        for n in range(N):
            n_nbest_hyps: List[Dict[str, Tensor]] = []
            for i, score in enumerate(nbest_scores[n]):
                new_hyp = {
                    "yseq": torch.tensor(nbest_ys[n, i, 1:nbest_ys_lengths[n, i]])
                }
                n_nbest_hyps.append(new_hyp)
            nbest_hyps.append(n_nbest_hyps)
        return nbest_hyps

    def eval(self):
        return self

    def cpu(self):
        return self

    def convert_ov_model(self, feats, lengths, beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):
        from openvino import save_model, convert_model

        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, feats, lengths):
                enc_outputs, _, enc_mask = self.model.encoder(feats, lengths)
        # "beam_size": 3,
        # "nbest": 1,
        # "decode_max_len": 0,
        # "softmax_smoothing": 1.0,
        # "aed_length_penalty": 0.0,
        # "eos_penalty": 1.0
                nbest_scores, nbest_ys, nbest_ys_lengths = self.model.decoder.batch_beam_search0(
                                            enc_outputs, enc_mask, beam_size, nbest=1,
                                            decode_max_len=0, softmax_smoothing=1.0,
                                            length_penalty=0.0, eos_penalty=1.0)
                return nbest_scores, nbest_ys, nbest_ys_lengths

        model = ModelWrapper(self.torch_model)
        model.eval()
        example_inputs = {"feats":feats, "lengths":lengths}
        ov_model = convert_model(model, example_input=example_inputs)
        save_model(ov_model, self.ov_path)
        
    def convert_ov_model1(self, feats, lengths, beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):
        from openvino import save_model, convert_model

        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, feats, lengths, beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty):
                enc_outputs, _, enc_mask = self.model.encoder(feats, lengths)
                nbest_scores, nbest_ys, nbest_ys_lengths = self.model.decoder.batch_beam_search0(
                                            enc_outputs, enc_mask, beam_size, nbest, decode_max_len,
                                            softmax_smoothing, length_penalty, eos_penalty)
                return nbest_scores, nbest_ys, nbest_ys_lengths

        model = ModelWrapper(self.torch_model)
        model.eval()
        print(f"feats={feats.shape}, lengths={lengths.shape}")
        example_inputs = {"feats":feats,
                          "lengths":lengths,
                          "beam_size": beam_size,
                          "nbest": nbest,
                          "decode_max_len": decode_max_len,
                          "softmax_smoothing": softmax_smoothing,
                          "length_penalty": length_penalty,
                          "eos_penalty": eos_penalty}
        ov_model = convert_model(model, example_input=example_inputs)
        save_model(ov_model, self.ov_path)