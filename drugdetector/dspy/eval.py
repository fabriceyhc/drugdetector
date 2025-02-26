import random
import os
import dspy
import pandas as pd
import traceback
import argparse
from dspy.datasets import DataLoader
from dspy.evaluate import Evaluate
from sglang.utils import launch_server_cmd, wait_for_server, print_highlight, terminate_process

from drugdetector.dspy.singlelabel.signatures import SinglelabelLE, SinglelabelEL, SinglelabelL
from drugdetector.dspy.multilabel.signatures import MultilabelLE, MultilabelEL, MultilabelL

# Constants
SEED = 0



# Dataset Preparation
def prepare_singlelabel_dataset(file_path, max_rows=None):
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

def prepare_multilabel_dataset(file_path, max_rows=None):
    df = pd.read_csv(file_path)

    dataset = [
        dspy.Example(
            medical_text=row["text"],
            heroin=row["heroin"],
            cocaine=row["cocaine"],
            methamphetamine=row["methamphetamine"],
            benzodiazepine=row["benzodiazepine"],
            rx_opioid_misuse=row["rx_opioid_misuse"],
            cannabis=row["cannabis"],
            injection_drug_use=row["injection_drug_use"],
            general_drug_use=row["general_drug_use"],
        ).with_inputs("medical_text")
        for _, row in df.iterrows()
    ]

    random.Random(SEED).shuffle(dataset)
    return dataset[:max_rows] if max_rows else dataset

# Score Metrics
def score_singlelabel(example, prediction, trace=None):
    return prediction.label == example.label

def score_multilabel(example, prediction, trace=None):
    # List all the label names as defined in your ClassifyL signature
    labels = [
        "heroin",
        "cocaine",
        "methamphetamine",
        "benzodiazepine",
        "rx_opioid_misuse",
        "cannabis",
        "injection_drug_use",
        "general_drug_use"
    ]
    
    # Count how many labels were predicted correctly
    correct = sum(getattr(example, label) == getattr(prediction, label) for label in labels)
    
    # Compute average accuracy across the 8 labels
    accuracy = correct / len(labels)
    return accuracy

# Evaluation Helper
def evaluate_program(program, model_id, signature_name, module_name, metric, num_demos, trainset, valset, testset):

    # Set save path and check if exists
    program_save_path = f'./drugdetector/dspy/prompts/{model_id}/MIPROv2_{module_name}-{signature_name}_demos={num_demos}.json'
    if os.path.exists(program_save_path):
        print(f"[EVAL] Found existing program; skipping!")
        # return {}

    os.makedirs(os.path.dirname(program_save_path), exist_ok=True)

    valset_evaluator = Evaluate(devset=valset, num_threads=1, display_progress=True)
    pre_score = valset_evaluator(program, metric=metric)
    
    optimizer = dspy.MIPROv2(
        metric=metric,
        auto="light",
        num_threads=24,
        max_labeled_demos=num_demos,
        max_bootstrapped_demos=num_demos
    )
    
    optimized_program = optimizer.compile(program, trainset=trainset, valset=valset, requires_permission_to_run=False)
    post_score = valset_evaluator(optimized_program, metric=metric)

    if pre_score >= post_score:
        print(f"[EVAL] pre_score={pre_score} is higher than post_score={post_score}; using original program!")
        program_to_eval = program
        original_program = True
    else:
        print(f"[EVAL] post_score={post_score} is higher than pre_score={pre_score}; using new program!")
        program_to_eval = optimized_program
        original_program = False

    program_to_eval.save(program_save_path, save_program=False)
    print(f"[EVAL] Program saved to {program_save_path}")

    print(f"[EVAL] Running saved program on testset...")
    testset_evaluator = Evaluate(devset=testset, num_threads=1, display_progress=True, return_outputs=True)
    overall_score, result_triples = valset_evaluator(optimized_program, metric=metric, return_outputs=True)
    test_results_df = process_result_triples(result_triples)

    test_save_path = f"./drugdetector/dspy/results/{model_id}/annotations_{module_name}-{signature_name}_demos={num_demos}.csv"
    os.makedirs(os.path.dirname(test_save_path), exist_ok=True)
    test_results_df.to_csv(test_save_path, index=False)
    print(f"[EVAL] Test results saved to {test_save_path}")

    return {
        "module": module_name,
        "signature": signature_name,
        "num_demos": num_demos,
        "pre_score": pre_score,
        "post_score": post_score,
        "overall_score": overall_score,
        "original_program": original_program,
        "program_save_path": program_save_path,
        "test_save_path": test_save_path
    }

def process_result_triples(result_triples):
    results = []
    for example, prediction, score in result_triples:
        example_dict = {k + '_true': v for k, v in example.toDict().items()}
        prediction_dict = {k + '_pred': v for k, v in prediction.toDict().items()}
        results.append({
            **example_dict,
            **prediction_dict,
            "score": score
        })
    df = pd.DataFrame(results)
    return df

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
            MultilabelL,
            MultilabelLE, MultilabelEL,
            SinglelabelL,
            SinglelabelLE, SinglelabelEL
        ]
        NUM_DEMOS_OPTIONS = [0, 3, 5, 10]

        # Experiment loop
        results = []
       
        for signature in SIGNATURES:
            signature_name = signature.__name__

            label_type = "multilabel" if "multi" in signature_name.lower() else "singlelabel"
            metric = score_multilabel if "multilabel" in label_type else score_singlelabel
            prepare_dataset = prepare_multilabel_dataset if "multilabel" in label_type else prepare_singlelabel_dataset
                
            # Data loading
            trainset = prepare_dataset(f"./data/{label_type}/train.csv", max_rows=10)
            valset   = prepare_dataset(f"./data/{label_type}/val.csv",   max_rows=10)
            testset  = prepare_dataset(f"./data/{label_type}/test.csv",  max_rows=10)
                
            for module in MODULES:
                program = module(signature)
                module_name = module.__name__
                
                for num_demos in NUM_DEMOS_OPTIONS:
                    result = evaluate_program(
                        program=program, 
                        model_id=model_id, 
                        signature_name=signature_name, 
                        module_name=module_name, 
                        metric=metric, 
                        num_demos=num_demos, 
                        trainset=trainset, 
                        valset=valset, 
                        testset=testset
                    )
                    results.append(result)

        # Ensure summary save directory exists
        summary_dir = f"./drugdetector/dspy/results/{model_id}"
        os.makedirs(summary_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(f"{summary_dir}/summary.csv")
        terminate_process(server_process)

    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        terminate_process(server_process)

if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=1 python -m drugdetector.dspy.eval --model_id meta-llama/Llama-3.1-8B-Instruct

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