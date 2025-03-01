# #!/bin/bash
# # Define common directories
# CACHE_DIR="/data2/.shared_models/llama.cpp_models"

# # List of model names to download
# models=(
# #   "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF"
# #   "bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF"
# #   "bartowski/OpenBioLLM-Llama3-8B-GGUF"
# #   "LiteLLMs/Llama3-OpenBioLLM-70B-GGUF"
# #   "mradermacher/Llama3-Med42-8B-GGUF"
# #   "tensorblock/Llama3-Med42-70B-GGUF"
# #   "featherless-ai-quants/ProbeMedicalYonseiMAILab-medllama3-v20-GGUF"
#   "MaziyarPanahi/Llama-3.3-70B-Instruct-GGUF"
# )

# # Loop through each model and run the download command
# for model in "${models[@]}"; do
#   huggingface-cli download "$model" \
#     --include "*Q8*.gguf" \
#     --cache-dir "$CACHE_DIR" \
#     --local-dir "$CACHE_DIR" \
#     --local-dir-use-symlinks False
# done

# # List of large models that require merging (directories with multiple shards)
# large_models=(
# #   "DeepSeek-R1-Distill-Llama-70B-Q8_0"
#   "Llama3-OpenBioLLM-70B-Q8_0"
#   "Llama3-Med42-70B-Q8_0"
#   "Llama-3.3-70B-Instruct-Q8_0"
# )

# # Loop through each large model and merge shards if necessary
# for model in "${large_models[@]}"; do
#   model_dir="$CACHE_DIR/$model"
#   # Look for shard files matching the pattern: <model>-*-of-*.gguf
#   shard_files=("$model_dir/${model}"-*-of-*.gguf)
  
#   if [ ${#shard_files[@]} -gt 1 ]; then
#     echo "Merging ${#shard_files[@]} shards for model $model"
#     /data2/fabricehc/llama.cpp/llama-gguf-split --merge "${shard_files[@]}" "$model_dir/${model}.gguf"
#   elif [ ${#shard_files[@]} -eq 1 ]; then
#     echo "Only one shard found for model $model; renaming it as merged file"
#     mv "${shard_files[0]}" "$model_dir/${model}.gguf"
#   else
#     echo "No shard files found for model $model"
#   fi
# done

# 8B Models

huggingface-cli download MaziyarPanahi/Llama-3.1-8B-Instruct-GGUF \
  --include "*Q8*.gguf" \
  --cache-dir /data2/.shared_models/ \
  --local-dir /data2/.shared_models/llama.cpp_models/ \
  --local-dir-use-symlinks False

huggingface-cli download lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF \
  --include "*Q8*.gguf" \
  --cache-dir /data2/.shared_models/ \
  --local-dir /data2/.shared_models/llama.cpp_models/ \
  --local-dir-use-symlinks False

huggingface-cli download bartowski/OpenBioLLM-Llama3-8B-GGUF \
  --include "*Q8*.gguf" \
  --cache-dir /data2/.shared_models/ \
  --local-dir /data2/.shared_models/llama.cpp_models/ \
  --local-dir-use-symlinks False

huggingface-cli download mradermacher/Llama3-Med42-8B-GGUF \
  --include "*Q8*.gguf" \
  --cache-dir /data2/.shared_models/ \
  --local-dir /data2/.shared_models/llama.cpp_models/ \
  --local-dir-use-symlinks False

huggingface-cli download featherless-ai-quants/ProbeMedicalYonseiMAILab-medllama3-v20-GGUF \
  --include "*Q8*.gguf" \
  --cache-dir /data2/.shared_models/ \
  --local-dir /data2/.shared_models/llama.cpp_models/ \
  --local-dir-use-symlinks False

# 70B Models

huggingface-cli download MaziyarPanahi/Llama-3.1-70B-Instruct-GGUF \
  --include "*Q8*.gguf" \
  --cache-dir /data2/.shared_models/ \
  --local-dir /data2/.shared_models/llama.cpp_models/ \
  --local-dir-use-symlinks False

huggingface-cli download MaziyarPanahi/Llama-3.3-70B-Instruct-GGUF \
  --include "*Q8*.gguf" \
  --cache-dir /data2/.shared_models/ \
  --local-dir /data2/.shared_models/llama.cpp_models/ \
  --local-dir-use-symlinks False

huggingface-cli download bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF \
  --include "*Q8*.gguf" \
  --cache-dir /data2/.shared_models/ \
  --local-dir /data2/.shared_models/llama.cpp_models/ \
  --local-dir-use-symlinks False

huggingface-cli download LiteLLMs/Llama3-OpenBioLLM-70B-GGUF \
  --include "*Q8*.gguf" \
  --cache-dir /data2/.shared_models/ \
  --local-dir /data2/.shared_models/llama.cpp_models/ \
  --local-dir-use-symlinks False

huggingface-cli download tensorblock/Llama3-Med42-70B-GGUF \
  --include "*Q8*.gguf" \
  --cache-dir /data2/.shared_models/ \
  --local-dir /data2/.shared_models/llama.cpp_models/ \
  --local-dir-use-symlinks False

# Merge 70B models

./llama-gguf-split --merge /data2/.shared_models/llama.cpp_models/Llama-3.1-70B-Instruct-Q8_0/Llama-3.1-70B-Instruct-Q8_0-00001-of-00002.gguf /data2/.shared_models/llama.cpp_models/Llama-3.1-70B-Instruct-Q8_0/Llama-3.1-70B-Instruct-Q8_0.gguf
./llama-gguf-split --merge /data2/.shared_models/llama.cpp_models/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct.Q8_0.gguf-00001-of-00006.gguf /data2/.shared_models/llama.cpp_models/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0.gguf
./llama-gguf-split --merge /data2/.shared_models/llama.cpp_models/DeepSeek-R1-Distill-Llama-70B-Q8_0/DeepSeek-R1-Distill-Llama-70B-Q8_0-00001-of-00002.gguf /data2/.shared_models/llama.cpp_models/DeepSeek-R1-Distill-Llama-70B-Q8_0/DeepSeek-R1-Distill-Llama-70B-Q8_0.gguf
./llama-gguf-split --merge /data2/.shared_models/llama.cpp_models/Llama3-OpenBioLLM-70B-Q8_0/Llama3-OpenBioLLM-70B-Q8_0-00001-of-00002.gguf /data2/.shared_models/llama.cpp_models/Llama3-OpenBioLLM-70B-Q8_0/Llama3-OpenBioLLM-70B-Q8_0.gguf
./llama-gguf-split --merge /data2/.shared_models/llama.cpp_models/Llama3-Med42-70B-Q8_0/Llama3-Med42-70B-Q8_0-00001-of-00003.gguf /data2/.shared_models/llama.cpp_models/Llama3-Med42-70B-Q8_0/Llama3-Med42-70B-Q8_0.gguf

