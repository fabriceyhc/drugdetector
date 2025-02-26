CUDA_VISIBLE_DEVICES=3,4,5,6 python -m drugdetector.dspy.eval --model_id meta-llama/Llama-3.1-70B-Instruct
CUDA_VISIBLE_DEVICES=3,4,5,6 python -m drugdetector.dspy.eval --model_id meta-llama/Llama-3.3-70B-Instruct
CUDA_VISIBLE_DEVICES=3,4,5,6 python -m drugdetector.dspy.eval --model_id deepseek-ai/DeepSeek-R1-Distill-Llama-70B
# CUDA_VISIBLE_DEVICES=3,4,5,6 python -m drugdetector.dspy.eval --model_id fabriceyhc/Meta-Llama-3-70B-Instruct-DrugDetection-v3
CUDA_VISIBLE_DEVICES=3,4,5,6 python -m drugdetector.dspy.eval --model_id m42-health/Llama3-Med42-70B
CUDA_VISIBLE_DEVICES=3,4,5,6 python -m drugdetector.dspy.eval --model_id aaditya/Llama3-OpenBioLLM-70B --chat_template ./drugdetector/dspy/chat_templates/llama-3-instruct.jinja