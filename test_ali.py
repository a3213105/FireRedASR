from fireredasr.models.fireredasr import FireRedAsr
import json
from pydub import AudioSegment
from pathlib import Path
import difflib
import os
import psutil

# input_file_path="examples/wav/4bR5h-ecBZg-all.wav"
# json_file_path="examples/wav/testwav.json"

input_file_path="tests/4bR5h-ecBZg-all.wav"
json_file_path="tests/testwav.json"


with open(json_file_path, 'r') as file:
    inputs = json.load(file)

audio = AudioSegment.from_file(input_file_path)

models=[]
test_case = [["tor", "ch"],
             ["f32", "f32"], ["f32", "bf16"], ["f32", "f16"],
             ["bf16", "f32"], ["bf16", "bf16"], ["bf16", "f16"],
             ["f16", "f32"], ["f16", "bf16"], ["f16", "f16"],
            ]
# test_case = [["tor", "ch"], ["f32", "f32"]]
test_case = [["bf16", "bf16"]]
for enc_type, dec_type in test_case:
    model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L", 
                                       enc_type=enc_type, dec_type=dec_type, cache_size=50)
    models.append([model, f"{enc_type}-{dec_type}"])

segments = inputs['gpu_meta']['vad_results']['segments']

process = psutil.Process(os.getpid())
max_rss = 0
max_vms = 0
for model, typename in models:
    total_rtf = 0.0
    for i in range(len(segments)):
        batch_uttid = segments[i]['segment_wav_id']
        # start_time = float(segments[i]['start_time']) * 1000
        # end_time = float(segments[i]['end_time']) * 1000
        # predict_result = segments[i]['predict_firered_asr_aed_text']
        batch_wav_path = f"tests/{batch_uttid}.wav"
        # batch_wav_path_ath = Path(batch_wav_path)
        
        # if not batch_wav_path_ath.exists():
        #     sliced_audio = audio[start_time:end_time]
        #     sliced_audio.export(batch_wav_path, format="wav")
        
        batch_uttid = [batch_uttid]
        batch_wav_path = [batch_wav_path]

        # rtf = []
        # result_text = []
        # similarity = []
        # rtf_output = ""
        # similarity_output = ""
        # print(f"tid:{i} {batch_uttid}\tpredict_result:{predict_result}", flush=True)
    # for model, typename in models:
        results = model.transcribe(
            batch_uttid,
            batch_wav_path,
            {
                "beam_size": 3,
                "nbest": 1,
                "decode_max_len": 0,
                "softmax_smoothing": 1.25,
                "aed_length_penalty": 0.6,
                "eos_penalty": 1.0,
                "decode_min_len": 0,
                "repetition_penalty": 1.0,
                "llm_length_penalty": 0.0,
                "temperature": 1.0
            }
        )
        # rtf.append(results[0]['rtf'])
        # result_text.append(results[0]['text'])
        # print(f"infer_mode:{typename}\trtf={results[0]['rtf']}, results= {results[0]['text']}")
        # sim = difflib.SequenceMatcher(None, results[0]['text'], predict_result).ratio()
        # print(f"{typename}={sim:.2f}", flush=True)
        # similarity_output += f", {typename}={sim:.2f}"
        # rtf_output += f", {typename}={results[0]['rtf']}"
        total_rtf += float(results[0]['rtf'])
        # print(f"infer_mode:{typename}\trtf={results[0]['rtf']}")
    
    total_rtf = total_rtf/len(segments)
    mem_info = process.memory_info()
    rss = mem_info.rss / 1024 ** 3
    vms = mem_info.vms / 1024 ** 3
    if max_rss < rss :
        max_rss = rss
    if max_vms < vms :
        max_vms = vms
    print(f"RSS: {rss:.2f} GB, VMS: {vms:.2f} GB, Max RSS: {max_rss:.2f} GB, Max VMS: {max_vms:.2f} GB, total_rtf={total_rtf}")  # 虚拟内存
    # print(f"tid:{i} {batch_uttid}\tsimilarity:{similarity_output}\trtf:{rtf_output}", flush=True)
