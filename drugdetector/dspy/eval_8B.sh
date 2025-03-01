CUDA_VISIBLE_DEVICES=0 python -m drugdetector.dspy.eval --model_id meta-llama/Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=0 python -m drugdetector.dspy.eval --model_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B
CUDA_VISIBLE_DEVICES=0 python -m drugdetector.dspy.eval --model_id ProbeMedicalYonseiMAILab/medllama3-v20
CUDA_VISIBLE_DEVICES=0 python -m drugdetector.dspy.eval --model_id m42-health/Llama3-Med42-8B
CUDA_VISIBLE_DEVICES=0 python -m drugdetector.dspy.eval --model_id aaditya/Llama3-OpenBioLLM-8B --chat_template ./drugdetector/dspy/chat_templates/llama-3-instruct.jinja
