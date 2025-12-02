from fireredasr.models.fireredasr import FireRedAsr

import os
import psutil

batch_uttid = ["1","2","3","4"]
batch_wav_path = ["examples/wav/BAC009S0764W0121.wav",
                  "examples/wav/IT0011W0001.wav",
                  "examples/wav/TEST_NET_Y0000000000_-KTKHdZ2fb8_S00000.wav",
                  "examples/wav/TEST_MEETING_T0000000001_S00000.wav"]

batch_uttid = ["1"]
batch_wav_path = ["examples/wav/BAC009S0764W0121.wav"]
# batch_wav_path = ["examples/wav/IT0011W0001.wav"]
# batch_wav_path = ["examples/wav/TEST_NET_Y0000000000_-KTKHdZ2fb8_S00000.wav"]
# batch_wav_path = ["examples/wav/TEST_MEETING_T0000000001_S00000.wav"]
# batch_wav_path = ["examples/wav/4bR5h-ecBZg.wav"]
# batch_wav_path = ["examples/wav/7184a192e882c87276778a96423a29c6_585.885.wav"]
# batch_wav_path = ["examples/wav/7184a192e882c87276778a96423a29c6_2.080.wav"]
batch_wav_path = ["examples/wav/7184a192e882c87276778a96423a29c6_835.655.wav"]
batch_wav_path = ["tests/7184a192e882c87276778a96423a29c6_393.625.wav"]

duration_list = [4.2, 1.99, 1.8, 12.37, 27.26]
results_list = ["甚至出现交易几乎停滞的情况", 
                "换一首歌", 
                "我有的时候说不清楚你们知道吗", 
                "好首先说一下刚才这个经理说完的这个销售问题咱再说一下咱们的商场问题首先咱们商场上半年业这个先各部门儿汇报一下就是业绩"]


# FireRedASR-AED
models=[]
model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L", enc_type="tor", dec_type="ch")
models.append([model, "torch"])
model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L", enc_type="f32", dec_type="f32")
models.append([model, "f32-f32"])
model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L", enc_type="f32", dec_type="bf16")
models.append([model, "f32-bf16"])
model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L", enc_type="f32", dec_type="f16")
models.append([model, "f32-f16"])
process = psutil.Process(os.getpid())
memory_info = process.memory_info()
rss_mb = memory_info.rss / (1024 * 1024)
vms_mb = memory_info.vms / (1024 * 1024)
print(f"RSS (常驻内存集): {rss_mb} MB")
print(f"VMS (虚拟内存集): {vms_mb} MB")

texts = []
for i, (model, typename) in enumerate(models):   
    results = model.transcribe(
        batch_uttid,
        batch_wav_path,
        {
            # "infer_mode": i,  # 0: torch, 1: ov_encoder, 2: ov_encoder+decoder, 3: ov_decoder
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

    print(f"infer_mode({i}):{typename}\trtf={results[0]['rtf']}, results= {results[0]['text']}")
    texts.append([results[0]['text'], results[0]['rtf']])

memory_info = process.memory_info()
rss_mb = memory_info.rss / (1024 * 1024)
vms_mb = memory_info.vms / (1024 * 1024)
print(f"RSS (常驻内存集): {rss_mb} MB")
print(f"VMS (虚拟内存集): {vms_mb} MB")

import difflib
for i, (_, typename) in enumerate(models):   
    similarity = difflib.SequenceMatcher(None, texts[i][0], texts[0][0]).ratio()
    print(f"similarity_{typename}={similarity:.2f}, rtf={texts[i][1]}")

