#!/bin/bash
# start llama.cpp server (qwen chat format, full GPU offload)
pkill -9 -f llama_cpp.server 2>/dev/null
sleep 2
cd ~
source llama-venv/bin/activate
setsid python3 -m llama_cpp.server \
  --model /home/zhong/models/qwen-coder-7b-q4.gguf \
  --host 0.0.0.0 --port 8001 \
  --n_ctx 8192 --n_gpu_layers 99 \
  --chat_format qwen \
  > ~/llama-server.log 2>&1 < /dev/null &
echo "launched pid $!"
sleep 18
tail -3 ~/llama-server.log
