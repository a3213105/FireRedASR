from fireredasr.models.fireredasr import FireRedAsr
import json
from pydub import AudioSegment
from pathlib import Path
import difflib
import os
import psutil
import glob

def load_dir(directory):
    segments = []
    wav_files = glob.glob(os.path.join(directory, "**", "*.*"), recursive=True)
    for file in wav_files:
        # print(f"load file: {file}")
        segments.append(file)
    return segments

def load_dir1(directory):
    segments = ["/home/sgui/test_audios/20.wav",
                "/home/sgui/test_audios/20.wav",
                "/home/sgui/test_audios/30.wav",
                "/home/sgui/test_audios/30.wav",
                "/home/sgui/test_audios/50.wav",
                "/home/sgui/test_audios/50.wav",
                "/home/sgui/test_audios/100.wav",
                "/home/sgui/test_audios/100.wav",
                "/home/sgui/test_audios/200.wav",
                "/home/sgui/test_audios/20.wav",]
    return segments

# segments = load_dir("/home/sgui/Ali/audios/audios")
segments = load_dir1("/home/sgui/test_audios")
# input_file_path="examples/wav/4bR5h-ecBZg-all.wav"
# json_file_path="examples/wav/testwav.json"

input_file_path="tests/4bR5h-ecBZg-all.wav"
json_file_path="tests/testwav.json"


# with open(json_file_path, 'r') as file:
#     inputs = json.load(file)
# segments = inputs['gpu_meta']['vad_results']['segments']

# audio = AudioSegment.from_file(input_file_path)

models=[]
# test_case = [["tor", "ch"],
#              ["f32", "f32"], ["f32", "bf16"], ["f32", "f16"],
#              ["bf16", "f32"], ["bf16", "bf16"], ["bf16", "f16"],
#              ["f16", "f32"], ["f16", "bf16"], ["f16", "f16"],
#             ]
# test_case = [["tor", "ch"], ["f32", "f32"]]
# test_case = [["bf16", "bf16"], ["f16", "f16"]]
test_case = [["bf16", "bf16"]]
for enc_type, dec_type in test_case:
    model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L", 
                                       enc_type=enc_type, dec_type=dec_type)
    models.append([model, f"{enc_type}-{dec_type}"])

mem = psutil.virtual_memory()
total = mem.total / 1024 ** 3
print(f"总内存: {total:.2f} GB")

max_rss = 0
max_vms = 0
process = psutil.Process(os.getpid())
mem_info = process.memory_info()
rss = mem_info.rss / 1024 ** 3
vms = mem_info.vms / 1024 ** 3
if max_rss < rss :
    max_rss = rss
if max_vms < vms :
    max_vms = vms
count = len(segments)
print(f"Init RSS: {rss:.2f} GB, VMS: {vms:.2f} GB, "
      f"Max RSS: {max_rss:.2f} GB, Max VMS: {max_vms:.2f} GB, "
      f"{max_rss / total * 100:.2f}%, {count} files to process")
for j in range(1):
    for model, typename in models:
        total_rtf = 0.0
        count = 10 #len(segments)
        count = len(segments)
        for i in range(count):
            batch_wav_path = segments[i]
                   
            batch_uttid = [i]
            batch_wav_path = [batch_wav_path]

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
            total_rtf += float(results[0]['rtf'])
            mem_info = process.memory_info()
            rss = mem_info.rss / 1024 ** 3
            vms = mem_info.vms / 1024 ** 3
            if max_rss < rss :
                max_rss = rss
            if max_vms < vms :
                max_vms = vms
            print(f"RSS: {rss:.2f} GB, VMS: {vms:.2f} GB, "
                  f"Max RSS: {max_rss:.2f} GB, Max VMS: {max_vms:.2f} GB, {max_rss / total * 100:.2f}%, "
                  f"batch_wav_path={batch_wav_path}"
                 )  
                
        total_rtf = total_rtf/count
        print(f"total_rtf={total_rtf:.2f} @ {count}")  
