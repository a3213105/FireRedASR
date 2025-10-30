import os
import time

import torch

from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.models.fireredasr_aed import FireRedAsrAed_ov, FireRedAsrAed
from fireredasr.models.fireredasr_llm import FireRedAsrLlm, FireRedAsrLlm_ov
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper


class FireRedAsr:
    @classmethod
    def from_pretrained(cls, asr_type, model_dir, infer_type):
        assert asr_type in ["aed", "llm"]

        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        feat_extractor = ASRFeatExtractor(cmvn_path)

        if asr_type == "aed":
            model_path = os.path.join(model_dir, "model.pth.tar")
            dict_path =os.path.join(model_dir, "dict.txt")
            spm_model = os.path.join(model_dir, "train_bpe1000.model")
            model = load_fireredasr_aed_model(model_path, infer_type)
            tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)
        elif asr_type == "llm":
            model_path = os.path.join(model_dir, "model.pth.tar")
            encoder_path = os.path.join(model_dir, "asr_encoder.pth.tar")
            llm_dir = os.path.join(model_dir, "Qwen2-7B-Instruct")
            model, tokenizer = load_firered_llm_model_and_tokenizer(
                model_path, encoder_path, llm_dir)
        model.eval()
        return cls(asr_type, feat_extractor, model, tokenizer)

    def __init__(self, asr_type, feat_extractor, model, tokenizer):
        self.asr_type = asr_type
        self.feat_extractor = feat_extractor
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def transcribe(self, batch_uttid, batch_wav_path, args={}):
        feats, lengths, durs = self.feat_extractor(batch_wav_path)
        total_dur = sum(durs)
        if args.get("use_gpu", False):
            feats, lengths = feats.cuda(), lengths.cuda()
            self.model.cuda()
        else:
            self.model.cpu()

        self.model.infer_mode = args.get("infer_mode", 0)
        if self.asr_type == "aed":
            beam_size = torch.tensor(args.get("beam_size", 1), dtype=torch.int32)
            nbest = torch.tensor(args.get("nbest", 1), dtype=torch.int32)
            decode_max_len = torch.tensor(args.get("decode_max_len", 0), dtype=torch.int32)
            softmax_smoothing = torch.tensor(args.get("softmax_smoothing", 1.0), dtype=torch.float32)
            length_penalty = torch.tensor(args.get("aed_length_penalty", 0.0), dtype=torch.float32)
            eos_penalty = torch.tensor(args.get("eos_penalty", 1.0), dtype=torch.float32)

            start_time = time.time()

            hyps = self.model.transcribe(feats, lengths, beam_size, nbest,
                                         decode_max_len, softmax_smoothing,
                                         length_penalty, eos_penalty)

            elapsed = time.time() - start_time
            rtf= elapsed / total_dur if total_dur > 0 else 0
            results = []
            for uttid, wav, hyp in zip(batch_uttid, batch_wav_path, hyps):
                hyp = hyp[0]  # only return 1-best
                hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
                text = self.tokenizer.detokenize(hyp_ids)
                results.append({"uttid": uttid, "text": text, "wav": wav,
                    "rtf": f"{rtf:.4f}", "elapsed": f"{elapsed:.3f}"})
            return results

        elif self.asr_type == "llm":
            input_ids, attention_mask, _, _ = \
                LlmTokenizerWrapper.preprocess_texts(
                    origin_texts=[""]*feats.size(0), tokenizer=self.tokenizer,
                    max_len=128, decode=True)
            if args.get("use_gpu", False):
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            start_time = time.time()

            generated_ids = self.model.transcribe(
                feats, lengths, input_ids, attention_mask,
                args.get("beam_size", 1),
                args.get("decode_max_len", 0),
                args.get("decode_min_len", 0),
                args.get("repetition_penalty", 1.0),
                args.get("llm_length_penalty", 0.0),
                args.get("temperature", 1.0)
            )

            elapsed = time.time() - start_time
            rtf= elapsed / total_dur if total_dur > 0 else 0
            texts = self.tokenizer.batch_decode(generated_ids,
                                                skip_special_tokens=True)
            results = []
            for uttid, wav, text in zip(batch_uttid, batch_wav_path, texts):
                results.append({"uttid": uttid, "text": text, "wav": wav,
                                "rtf": f"{rtf:.4f}"})
            return results


def load_fireredasr_aed_model(model_path, infer_type):
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_ov = FireRedAsrAed_ov(package["args"], None, model_path, infer_type)
    model_ov.using_ov = False
    if not model_ov.using_ov:
        model = FireRedAsrAed.from_args(package["args"])
        model.load_state_dict(package["model_state_dict"], strict=True)
        model.eval()
        model_ov.torch_model = model
    return model_ov


def load_firered_llm_model_and_tokenizer(model_path, encoder_path, llm_dir):
    model_ov = FireRedAsrLlm_ov(None, model_path, infer_type="bf16")
    # model_ov.using_ov = False
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    package["args"].encoder_path = encoder_path
    package["args"].llm_dir = llm_dir
    print("model args:", package["args"])
    model = FireRedAsrLlm.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    model_ov.llm_model = model
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(llm_dir)
    return model_ov, tokenizer
