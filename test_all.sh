#!/usr/bin/bash

test_cases=(
  "0 tor ch" "1 tor ch"
  "0 f32 f32" "1 f32 f32" "2 f32 f32"
  "0 bf16 bf16" "1 bf16 bf16" "2 bf16 bf16"
  "0 f16 f16" "1 f16 f16" "2 f16 f16"
)

for row in "${test_cases[@]}"; do
  # 将 row 按空格拆分为 3 个字段
  IFS=' ' read -r implement_type enc_type dec_type <<< "${row}"
#   echo "implement_type=${implement_type}, enc_type=${enc_type}, dec_type=${dec_type}"
  numactl -C32-39,160-167 python test_ali.py -i ${implement_type} -e ${enc_type} -d ${dec_type}
done
