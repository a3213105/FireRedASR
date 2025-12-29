from fireredasr.models.fireredasr import FireRedAsr
import json
from pydub import AudioSegment
from pathlib import Path
import difflib
import os
import psutil
import glob

import gc

check_mem=False
sample_type=2
test_type=5

def load_dir(directory):
    segments = []
    wav_files = glob.glob(os.path.join(directory, "**", "*.*"), recursive=True)
    for file in wav_files:
        segments.append(file)
    return segments

def load_dir1(directory):
    segments = ["/home/sgui/test_audios/100.wav",
                "/home/sgui/test_audios/20.wav",
                "/home/sgui/test_audios/100.wav",
                "/home/sgui/test_audios/200.wav",
                "/home/sgui/test_audios/30.wav",
                "/home/sgui/test_audios/200_1.wav",
                "/home/sgui/test_audios/100.wav",
                "/home/sgui/test_audios/20.wav",
                "/home/sgui/test_audios/200_2.wav",
                "/home/sgui/test_audios/100.wav",
                "/home/sgui/test_audios/50.wav",
                "/home/sgui/test_audios/200_3.wav",
                "/home/sgui/test_audios/20.wav",
                "/home/sgui/test_audios/200_4.wav",
                "/home/sgui/test_audios/100.wav",
                "/home/sgui/test_audios/200_5.wav",
                "/home/sgui/test_audios/20.wav",
                "/home/sgui/test_audios/200_6.wav",
                "/home/sgui/test_audios/20.wav",
                "/home/sgui/test_audios/30.wav",
                "/home/sgui/test_audios/50.wav",
                ]
    segments = ["/home/sgui/test_audios/20.wav",
                "/home/sgui/test_audios/100.wav",
                "/home/sgui/test_audios/20.wav",]
    # segments = ["/home/sgui/test_audios/100.wav",]
    return segments

if sample_type==0:
    segments = load_dir1("/home/sgui/test_audios")
elif sample_type==1:
    segments = load_dir("/home/sgui/test_audios")
else :
    input_file_path="tests/4bR5h-ecBZg-all.wav"
    json_file_path="tests/testwav.json"
    with open(json_file_path, 'r') as file:
        inputs = json.load(file)
    audio = AudioSegment.from_file(input_file_path)
    segments = []
    for segment in inputs['gpu_meta']['vad_results']['segments'] :
        segment_wav_id = segment['segment_wav_id']
        start_time = float(segment['start_time']) * 1000
        end_time = float(segment['end_time']) * 1000
        batch_wav_path = f"tests/{segment_wav_id}.wav"
        if not os.path.exists(batch_wav_path):
            sliced_audio = audio[start_time:end_time]
            sliced_audio.export(batch_wav_path, format="wav")
        segments.append(batch_wav_path)

# segments = ["/home/sgui/test_audios/121-123852-0004.flac",
#             "/home/sgui/test_audios/121-123852-0004.flac"]
models=[]
# test_case = [["tor", "ch"],
#              ["f32", "f32"], ["f32", "bf16"], ["f32", "f16"],
#              ["bf16", "f32"], ["bf16", "bf16"], ["bf16", "f16"],
#              ["f16", "f32"], ["f16", "bf16"], ["f16", "f16"],
#             ]
# test_case = [["tor", "ch"], ["f32", "f32"]]
# test_case = [["bf16", "bf16"], ["f16", "f16"]]
if test_type==0:
    test_case = [[0, "bf16", "bf16"],[1, "bf16", "bf16"],[2, "bf16", "bf16"]]
elif test_type==1:
    test_case = [["bf16", "bf16"], ["f16", "f16"]]
elif test_type==2:
    test_case = [["f32", "f32"], ["bf16", "bf16"]]
elif test_type==3:
    test_case = [["f32", "f32"], ["bf16", "bf16"], ["f16", "f16"]]
elif test_type==4:
    test_case = [["tor", "ch"], ["f32", "f32"], ["bf16", "bf16"], ["f16", "f16"]]
elif test_type==5:
    test_case = [[0, "tor", "ch"],
                 [0, "f32", "f32"],[1, "f32", "f32"],[2, "f32", "f32"],
                 [0, "bf16", "bf16"],[1, "bf16", "bf16"],[2, "bf16", "bf16"],
                 [0, "f16", "f16"],[1, "f16", "f16"],[2, "f16", "f16"]]

for implement_type, enc_type, dec_type in test_case:
    model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L", 
                                       implement_type=implement_type,
                                       enc_type=enc_type, dec_type=dec_type)
    models.append([model, f"{implement_type}-{enc_type}-{dec_type}"])

if check_mem:
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
            if len(results[0]['text'])==0:
                print(f"{typename}, {segments[i]}")
            total_rtf += float(results[0]['rtf'])
            if check_mem:
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
        print(f"{typename}, total_rtf={total_rtf:.2f} @ {count}")  
