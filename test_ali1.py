from fireredasr.models.fireredasr import FireRedAsr
import json
from pydub import AudioSegment
from pathlib import Path
import difflib
import os
import psutil
import glob
import argparse

import gc

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

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('Test FireRedASR', add_help=False)
    parser.add_argument('--sample', '-s', default=2, type=int, help='input sample Type (0, 1, 2)')
    parser.add_argument('--implement', '-i', default=2, type=int, help='implement Type (0, 1, 2)')
    parser.add_argument('--enc', '-e', default='bf16', type=str, help='encoder type: Torch, F32, BF16, F16')
    parser.add_argument('--dec', '-d', default='bf16', type=str, help='encoder type: Torch, F32, BF16, F16')
    parser.add_argument('--warmup', '-w', default=1, type=int, help='warmup iterations')
    parser.add_argument('--loop', '-l', default=0, type=int, help='loop iterations')
    parser.add_argument('--check_mem', '-m', type=str2bool, default=False, help="Check memory usage")
    return parser

parser = get_args_parser()
args = parser.parse_args()

if args.sample==0:
    segments = load_dir1("/home/sgui/test_audios")
elif args.sample==1:
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

model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L", 
                                   implement_type=args.implement, enc_type=args.enc, dec_type=args.dec)
model_name = f"{args.implement}-{args.enc}-{args.dec}"

if args.check_mem:
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

total_count = len(segments)
if total_count < args.warmup:
    args.warmup = total_count
elif args.warmup < 0:
    args.warmup = 0

if total_count < args.loop or args.loop < 1:
    args.loop = total_count

for index in range(args.warmup):
    batch_wav_path = segments[index]                 
    batch_uttid = [index]
    batch_wav_path = [batch_wav_path]          
    results = model.transcribe(batch_uttid, batch_wav_path,
            {"beam_size": 3, "nbest": 1, "decode_max_len": 0,
             "softmax_smoothing": 1.25, "aed_length_penalty": 0.6,
             "eos_penalty": 1.0, "decode_min_len": 0,
             "repetition_penalty": 1.0, "llm_length_penalty": 0.0, 
             "temperature": 1.0})

for j in range(1):
    total_rtf = 0.0
    total_dur = 0.0
    total_elapsed = 0.0
    total_tokens = 0
    # total_count = 3
    for i in range(args.loop):
        batch_wav_path = segments[i]
                   
        batch_uttid = [i]
        batch_wav_path = [batch_wav_path]
            
        results = model.transcribe(
            batch_uttid, batch_wav_path,
            {"beam_size": 3, "nbest": 1, "decode_max_len": 0,
             "softmax_smoothing": 1.25, "aed_length_penalty": 0.6,
             "eos_penalty": 1.0, "decode_min_len": 0,
             "repetition_penalty": 1.0, "llm_length_penalty": 0.0,
             "temperature": 1.0})
        if len(results[0]['text'])==0:
            print(f"{model_name}, {segments[i]}")
        total_rtf += float(results[0]['rtf'])
        total_dur += float(results[0]['total_dur'])
        total_elapsed += float(results[0]['elapsed'])
        total_tokens += int(results[0]['tokens'])
        if args.check_mem:
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
    total_rtf = total_rtf/args.loop
    total_rtf1 = total_elapsed / total_dur
    print(f"{model_name}, total_rtf={total_rtf:.4f}, total_duration={total_dur}, "
          f"total_processing_latency={total_elapsed:.4f}, total_tokens={total_tokens}, "
          f"rtf={total_rtf1:.4f} @ {args.loop}")  
