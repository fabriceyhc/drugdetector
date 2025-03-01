"""
Drug Detection Evaluation Pipeline

This script evaluates different DSPy configurations for drug detection classification tasks.
"""

# Standard library
import argparse
import os
import random
import traceback
from functools import partial
from typing import List, Dict, Any, Optional, Tuple

# Third-party
import dspy
import pandas as pd
from dspy.evaluate import Evaluate
from sglang.utils import launch_server_cmd, wait_for_server, terminate_process

# Local modules
from drugdetector.dspy.settings.singlelabel.signatures import (
    SinglelabelEL, SinglelabelL
)
from drugdetector.dspy.settings.multilabel.signatures import (
    MultilabelEL, MultilabelL
)

# Constants
SEED = 0
BASE_DATA_PATH = "./data"
BASE_SAVE_PATH = "./drugdetector/dspy"
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
LABELS = [
    "heroin",
    "cocaine",
    "methamphetamine",
    "benzodiazepine",
    "rx_opioid_misuse",
    "cannabis",
    "injection_drug_use",
    "general_drug_use"
]


def prepare_dataset(file_path: str, 
                   max_rows: Optional[int] = None,
                   dataset_type: str = "singlelabel") -> List[dspy.Example]:
    """Prepares dataset for drug detection tasks.
    
    Args:
        file_path: Path to CSV data file
        max_rows: Maximum number of rows to load
        dataset_type: 'singlelabel' or 'multilabel' format
    
    Returns:
        List of shuffled dspy.Example instances
    """
    df = pd.read_csv(file_path)
    
    if dataset_type == "singlelabel":
        df['drug_description'] = df['drug'].map(DRUG_DESCRIPTIONS)
        examples = [
            dspy.Example(
                medical_text=row["text"],
                drug_name=row["drug"],
                drug_description=row["drug_description"],
                label=row["label"]
            ).with_inputs("medical_text", "drug_name", "drug_description")
            for _, row in df.iterrows()
        ]
    else:
        examples = [
            dspy.Example(
                medical_text=row["text"],
                **{label: row[label] for label in LABELS}
            ).with_inputs("medical_text")
            for _, row in df.iterrows()
        ]

    random.Random(SEED).shuffle(examples)
    return examples[:max_rows] if max_rows else examples


def score_singlelabel(example: dspy.Example, 
                     prediction: dspy.Example, 
                     trace: Any = None) -> bool:
    """Calculates accuracy for single-label classification."""
    return prediction.label == example.label


def score_multilabel(example: dspy.Example, 
                    prediction: dspy.Example, 
                    trace: Any = None) -> float:
    """Calculates multilabel accuracy across all drug categories."""
    correct = sum(getattr(example, label) == getattr(prediction, label)
                for label in LABELS)
    return correct / len(LABELS)


def evaluate_program(program: dspy.Module,
                   model_id: str,
                   signature_name: str,
                   module_name: str,
                   metric: callable,
                   num_demos: int,
                   trainset: List[dspy.Example],
                   valset: List[dspy.Example],
                   testset: List[dspy.Example]) -> Dict[str, Any]:
    """Evaluates a DSPy program configuration.
    
    Returns:
        Dictionary containing evaluation results and metadata
    """
    # Path setup
    program_save_path = os.path.join(
        BASE_SAVE_PATH, "programs", model_id,
        f"MIPROv2_{module_name}-{signature_name}_demos={num_demos}.json"
    )
    if os.path.exists(program_save_path):
        print(f"[EVAL] Found existing program at {program_save_path}; skipping!")
        return {}

    os.makedirs(os.path.dirname(program_save_path), exist_ok=True)

    # Evaluation workflow
    val_evaluator = Evaluate(devset=valset, num_threads=1, display_progress=True)
    pre_score = val_evaluator(program, metric=metric)
    
    optimizer = dspy.MIPROv2(
        metric=metric,
        auto="medium",
        num_threads=24,
        max_labeled_demos=num_demos,
        max_bootstrapped_demos=num_demos,
        verbose=True,
    )
    
    optimized_program = optimizer.compile(
        student=program.deepcopy(), 
        trainset=trainset, 
        valset=valset, 
        requires_permission_to_run=False
    )

    print(optimized_program)

    # assert len(optimized_program.demos) == num_demos

    post_score = val_evaluator(optimized_program, metric=metric)

    # Save and test
    optimized_program.save(program_save_path, save_program=False)
    
    test_save_path = os.path.join(
        BASE_SAVE_PATH, "results", model_id,
        f"annotations_{module_name}-{signature_name}_demos={num_demos}.csv"
    )
    os.makedirs(os.path.dirname(test_save_path), exist_ok=True)
    
    test_evaluator = Evaluate(devset=testset, num_threads=1, 
                            display_progress=True, return_outputs=True)
    overall_score, result_triples = test_evaluator(
        optimized_program, metric=metric, return_outputs=True
    )
    
    # Process and save results
    results = []
    for example, prediction, score in result_triples:
        record = {
            **{f"{k}_true": v for k, v in example.toDict().items()},
            **{f"{k}_pred": v for k, v in prediction.toDict().items()},
            "score": score
        }
        results.append(record)
    
    pd.DataFrame(results).to_csv(test_save_path, index=False)
    print(f"[EVAL] Test results saved to {test_save_path}")

    return {
        "module": module_name,
        "signature": signature_name,
        "num_demos": num_demos,
        "pre_score": pre_score,
        "post_score": post_score,
        "overall_score": overall_score,
        "program_path": program_save_path,
        "test_results_path": test_save_path
    }


import json
import tempfile

class ServerManager:
    """Context manager for SGLang server lifecycle."""
    
    def __init__(self, model_id: str, chat_template: str):
        self.model_id = model_id
        self.chat_template = chat_template
        self.process = None
        self.port = None
        self.temp_file = None

    def __enter__(self):
        # GPU configuration
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        num_gpus = len([d for d in cuda_devices.split(",") if d.strip()]) or 1

        # Server launch
        server_cmd = (
            f"python -m sglang.launch_server --model-path {self.model_id} "
            f"--download-dir /data2/.shared_models/hf --tp {num_gpus}"
        )

        if self.chat_template != "default":
            server_cmd += f" --chat-template {self.chat_template}"
        
        self.process, self.port = launch_server_cmd(server_cmd)
        wait_for_server(f"http://localhost:{self.port}")
        print(f"Server running on port {self.port}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup
        if self.temp_file and os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        terminate_process(self.process)
        print("Server terminated")


class ExperimentConfig:
    """Central configuration for experiment parameters."""
    
    def __init__(self):
        self.modules = [dspy.Predict, dspy.ChainOfThought]
        self.signatures = [
            (MultilabelL, "multilabel"),
            (MultilabelEL, "multilabel"),
            (SinglelabelL, "singlelabel"),
            (SinglelabelEL, "singlelabel"),
        ]
        self.demo_options = [0] # [0, 3, 5, 10]


def main(model_id: str, chat_template: str) -> None:
    """Main execution flow."""
    config = ExperimentConfig()
    
    with ServerManager(model_id, chat_template) as server:
        try:
            # Model setup
            lm = dspy.LM(
                f"openai/{model_id}",
                api_base=f"http://localhost:{server.port}/v1",
                api_key="local",
                model_type='chat'
            )
            dspy.configure(lm=lm)

            results = []
            
            for Signature, label_type in config.signatures:
                # Dataset loading
                data_path = partial(
                    os.path.join,
                    BASE_DATA_PATH, 
                    label_type
                )
                prepare_fn = partial(
                    prepare_dataset,
                    dataset_type=label_type,
                    # max_rows=30
                )

                # trainset = prepare_fn(data_path("train.csv"), max_rows=10)
                # valset = prepare_fn(data_path("val.csv"), max_rows=10)
                # testset = prepare_fn(data_path("test.csv"), max_rows=10)

                trainset = prepare_fn(data_path("train.csv"))
                valset = prepare_fn(data_path("val.csv"))
                testset = prepare_fn(data_path("test.csv"))
                
                # Module evaluation
                for Module in config.modules:
                    program = Module(Signature)
                    
                    for num_demos in config.demo_options:
                        result = evaluate_program(
                            program=program.deepcopy(),
                            model_id=model_id,
                            signature_name=Signature.__name__,
                            module_name=Module.__name__,
                            metric=score_multilabel if "multi" in label_type else score_singlelabel,
                            num_demos=num_demos,
                            trainset=trainset,
                            valset=valset,
                            testset=testset
                        )
                        if result:  # Skip empty results from cached runs
                            results.append(result)

            # Save summary
            summary_path = os.path.join(
                BASE_SAVE_PATH, "results", model_id, "summary.csv"
            )
            pd.DataFrame(results).to_csv(summary_path, index=False)
            print(f"\nExperiment summary saved to {summary_path}")

        except Exception as e:
            print(f"\nCritical error occurred: {str(e)}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate drug detection models using DSPy configurations."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model identifier"
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default="default",
        help="File path to a chat template in jinja format. Default means use the format in tokenizer. Some models don't have a template specified."
    )
    args = parser.parse_args()
    
    main(args.model_id, args.chat_template)