CUDA_VISIBLE_DEVICES=1 python -m drugdetector.dspy.singlelabel.annotate --model_id meta-llama/Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=1 python -m drugdetector.dspy.multilabel.annotate  --model_id meta-llama/Llama-3.1-8B-Instruct

CUDA_VISIBLE_DEVICES=1 python -m drugdetector.dspy.singlelabel.annotate --model_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B
CUDA_VISIBLE_DEVICES=1 python -m drugdetector.dspy.multilabel.annotate  --model_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# CUDA_VISIBLE_DEVICES=1,2,3,4 python -m drugdetector.dspy.singlelabel.annotate --model_id meta-llama/Llama-3.1-70B-Instruct
# CUDA_VISIBLE_DEVICES=1,2,3,4 python -m drugdetector.dspy.multilabel.annotate  --model_id meta-llama/Llama-3.1-70B-Instruct

# CUDA_VISIBLE_DEVICES=1,2,3,4 python -m drugdetector.dspy.singlelabel.annotate --model_id deepseek-ai/DeepSeek-R1-Distill-Llama-70B
# CUDA_VISIBLE_DEVICES=1,2,3,4 python -m drugdetector.dspy.multilabel.annotate  --model_id deepseek-ai/DeepSeek-R1-Distill-Llama-70B