import random
import os
import dspy
import pandas as pd
import textwrap
import traceback
import argparse
from dspy.datasets import DataLoader
from dspy.evaluate import Evaluate
from sglang.utils import launch_server_cmd, wait_for_server, print_highlight, terminate_process

# Constants
SEED = 0
DOC_STRING = """
Determine whether the medical text contains reference to illicit drug use.
Special Notes:
    1. The mere mention of a drug is not sufficient. You are only looking for illicit use of the drug in the medical note. Do not assume that a drug is being used illicitly without some evidence. 
    2. If the text warns against the use of a particular drug, that does not mean the patient is actually using the drug. 
    3. If family drug use is present, that is not relevant to the patient and should not be flagged. 
    4. If the patient denies using a particular drug, do not mark that drug as being present. For example, if the note says "patient denied using heroin", then the label should be False. 
    5. Many opioids and benzodiazepines are appropriately used and should not be noted. We only want you to identify cases where the patient is not using them appropriately. For example, if they are taking Percocets acquired from friends or from the streets, this would be considered illicit misuse. 
    6. Medical recommendations about drugs do not mean the patient is actually using the drug.          
"""
MEDICAL_TEXT = "A medical text containing potentially illicit drug use by a patient."
DRUG_NAME = "The name of the illicit drug to detect."
DRUG_DESC = "A description of the illicit drug."
LABEL = "Whether the illicit drug is detected in the medical note."
EXPLANATION = "Explanation for the detection status."

DRUG_DESCRIPTIONS = {
    "heroin": "Heroin is an illegal opioid drug known for its high potential for addiction and overdose.",
    "cocaine": "Cocaine is a powerful stimulant drug that is often abused for its euphoric effects.",
    "methamphetamine": "Methamphetamine (including illicit amphetamine use, but not prescribed amphetamines for ADHD) is a potent central nervous system stimulant that is highly addictive.",
    "benzodiazepine": "Benzodiazepines are a class of psychoactive drugs commonly prescribed for anxiety, insomnia, and other conditions but can be abused for their sedative effects.",
    "rx_opioid_misuse": "Prescription opioids (only if being misused or used illicitly, not if taken as prescribed) are medications typically prescribed for pain relief but can be highly addictive when misused.",
    "cannabis": "Cannabis, also known as marijuana, is often used recreationally or medicinally but can be illegal depending on the jurisdiction.",
    "injection_drug_use": "Injection drug use (IDU, IVDA, IVDU) refers to the use of drugs administered via needles, often associated with higher risks of infectious diseases.",
    "general_drug_use": "General drug use refers to the use of any illegal or illicit substances.",
}


class ClassifyLE(dspy.Signature):
    DOC_STRING
    medical_text: str = dspy.InputField(desc=MEDICAL_TEXT)
    drug_name: str = dspy.InputField(desc=DRUG_NAME)
    drug_description: str = dspy.InputField(desc=DRUG_DESC)
    explanation: str = dspy.OutputField(desc=EXPLANATION)
    label: bool = dspy.OutputField(desc=LABEL)

class ClassifyEL(dspy.Signature):
    DOC_STRING
    medical_text: str = dspy.InputField(desc=MEDICAL_TEXT)
    drug_name: str = dspy.InputField(desc=DRUG_NAME)
    drug_description: str = dspy.InputField(desc=DRUG_DESC)
    explanation: str = dspy.OutputField(desc=EXPLANATION)
    label: bool = dspy.OutputField(desc=LABEL)

class ClassifyL(dspy.Signature):
    DOC_STRING
    medical_text: str = dspy.InputField(desc=MEDICAL_TEXT)
    drug_name: str = dspy.InputField(desc=DRUG_NAME)
    drug_description: str = dspy.InputField(desc=DRUG_DESC)
    label: bool = dspy.OutputField(desc=LABEL)



def score_detection(example, prediction, trace=None):
    return prediction.label == example.label


# Dataset Preparation
def prepare_dataset(file_path, max_rows=None):
    df = pd.read_csv(file_path)
    df['drug_description'] = df['drug'].map(DRUG_DESCRIPTIONS)

    dataset = [
        dspy.Example(
            medical_text=row["text"],
            drug_name=row["drug"],
            drug_description=row["drug_description"],
            label=row["label"]
        ).with_inputs("medical_text", "drug_name", "drug_description")
        for _, row in df.iterrows()
    ]

    random.Random(SEED).shuffle(dataset)
    return dataset[:max_rows] if max_rows else dataset

# Evaluation Helper
def evaluate_model(evaluator, model, signature_name, module_name, num_demos, trainset, model_id):
    pre_score = evaluator(model, metric=score_detection)
    
    optimizer = dspy.MIPROv2(
        metric=score_detection,
        num_threads=24,
        max_labeled_demos=num_demos,
        max_bootstrapped_demos=num_demos
    )
    
    optimized_model = optimizer.compile(model, trainset=trainset, requires_permission_to_run=False)
    post_score = evaluator(optimized_model, metric=score_detection)

    save_path = f'./drugdetection/dspy/singlelabel/optimized_prompts/{model_id}/MIPROv2_{module_name}-{signature_name}_demos={num_demos}.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    optimized_model.save(save_path, save_program=False)
    
    return {
        "module": module_name,
        "signature": signature_name,
        "num_demos": num_demos,
        "pre_score": pre_score,
        "post_score": post_score,
        "save_path": save_path,
    }

# Main Execution
def main(model_id):

    # Determine the number of GPUs from CUDA_VISIBLE_DEVICES and set the --tp parameter
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    num_gpus = 1
    if cuda_visible_devices:
        # Count the non-empty entries (in case of stray commas)
        gpu_list = [gpu.strip() for gpu in cuda_visible_devices.split(",") if gpu.strip()]
        num_gpus = len(gpu_list)

    # Server setup with optional --tp argument
    server_cmd = f"python -m sglang.launch_server --model-path {model_id} --download-dir /data2/.shared_models/hf --tp {num_gpus}"
    server_process, port = launch_server_cmd(server_cmd)
    wait_for_server(f"http://localhost:{port}")
    print(f"SGLang server started on http://localhost:{port}")

    try:
        # Data loading
        trainset = prepare_dataset("./data/singlelabel/val.csv")
        testset  = prepare_dataset("./data/singlelabel/test.csv", max_rows=200)

        print(testset)

        # Model setup
        lm = dspy.LM(
            f"openai/{model_id}",
            api_base=f"http://localhost:{port}/v1",
            api_key="local",
            model_type='chat'
        )
        dspy.configure(lm=lm)

        MODULES = [dspy.Predict, dspy.ChainOfThought]
        SIGNATURES = [
            ClassifyL,
            ClassifyLE, ClassifyEL
        ]
        NUM_DEMOS_OPTIONS = [0, 3, 5, 10]

        # Experiment loop
        results = []
        evaluator = Evaluate(devset=testset, num_threads=1, display_progress=True)
        
        for signature in SIGNATURES:
            signature_name = signature.__name__
            
            for module in MODULES:
                model = module(signature)
                module_name = module.__name__
                
                for num_demos in NUM_DEMOS_OPTIONS:
                    result = evaluate_model(evaluator, model, signature_name, module_name, num_demos, trainset, model_id)
                    results.append(result)

        # Ensure summary save directory exists
        summary_dir = f"./story_eval/dspy/singlelabel/optimized_prompts/{model_id}"
        os.makedirs(summary_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(f"{summary_dir}/summary.csv")
        terminate_process(server_process)

    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        terminate_process(server_process)

if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=1 python -m story_eval.dspy.singlelabel.annotate --model_id meta-llama/Llama-3.1-8B-Instruct

    parser = argparse.ArgumentParser(
        description="Evaluate a model using SGLang for singlelabel tasks with a specified Hugging Face model_id."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model id to use (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    args = parser.parse_args()
    main(args.model_id)