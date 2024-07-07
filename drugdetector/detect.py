import warnings
warnings.filterwarnings("ignore")

import time
import traceback
import pandas as pd
import guidance
from guidance import models, gen, select, user, system, assistant
from drugdetector.utils import convert_bools_in_dict


n = "\n"

class DrugDetector:
    def __init__(self, 
                 model_id='TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ', 
                 cache_dir=None, 
                 device_map="auto",
                 drugs=None,
    ):       
        self.model = models.Transformers(
            model_id, 
            echo=False,
            cache_dir=cache_dir, 
            device_map=device_map
        )
        self.drugs = drugs
        if self.drugs is None:
            self.drugs = {
                "Heroin": "Heroin is an illegal opioid drug known for its high potential for addiction and overdose.",
                "Cocaine": "Cocaine is a powerful stimulant drug that is often abused for its euphoric effects.",
                "Methamphetamine": "Methamphetamine (including illicit amphetamine use, but not prescribed amphetamines for ADHD) is a potent central nervous system stimulant that is highly addictive.",
                "Benzodiazepine": "Benzodiazepines are a class of psychoactive drugs commonly prescribed for anxiety, insomnia, and other conditions but can be abused for their sedative effects.",
                "Prescription Opioids": "Prescription opioids (only if being misused or used illicitly, not if taken as prescribed) are medications typically prescribed for pain relief but can be highly addictive when misused.",
                "Cannabis": "Cannabis, also known as marijuana, is often used recreationally or medicinally but can be illegal depending on the jurisdiction.",
                "Injection Drugs": "Injection drug use (IDU, IVDA, IVDU) refers to the use of drugs administered via needles, often associated with higher risks of infectious diseases.",
                "General Drugs": "General drug use refers to the use of any illegal or illicit substances.",
            }

    def detect(self, medical_text, drugs=None, persona=None, examples=[], explain=False):

        if isinstance(drugs, dict):
            self.drugs = drugs
                
        @guidance(dedent=False)
        def annotation_task(lm, medical_text, persona, examples):
            if persona:
                with system():
                    lm += f"{persona}"

            with user():
                lm += f"""\
                ### Task Description:
                Please carefully review the following medical note for any mentions of drug use. 
                Specifically look for mentions of the following drugs:
                {f'{n}'.join([f'{drug}: {description}' for drug, description in self.drugs.items()])}

                NOTE: 
                - If the patient denies using a particular drug, do not mark that drug as being present. For example, if the note says "patient denied using heroin", then set "Heroin" to false.
                - If there are no mentions of any drug use whatsoever, set "General Drug Use" to false and do not mark any of the other substances as true.
                """

                # Adding few-shot examples if any are provided
                if examples:
                    lm += "\n### Examples:\n"
                    for example in examples:
                        lm += f"""
                        Example Medical Note: {example['text']}
                        
                        Feedback: 
                        {self._generate_examples(examples, self.drugs)}
                        """
                        
                lm += f"""\

                ### The medical text to evaluate:
                {medical_text}
                """
            with assistant():
                lm += f"""\
                ### Feedback: 
                {self._generate_explanations_and_select_options(self.drugs, explain)}
                """
            return lm

        # Execute the annotation task with the model
        try:
            start_time = time.time()
            output = self.model + annotation_task(medical_text, persona, examples)
            time_taken = time.time() - start_time
            results = convert_bools_in_dict({d: output[d] for d in self.drugs.keys()})
            results.update({
                "medical_text": medical_text,
                "time_taken": time_taken,
            })
            if explain:
                explanations = {d: f"{output[f'{d}_explanation'].strip()}." for d in self.drugs.keys()}
                results.update({f'{d}_explanation': explanations[d] for d in self.drugs.keys()})
            return results
        except Exception as e:
            print(f"An error occurred during annotation: {e}")
            traceback.print_exc()
            return None

    def _generate_examples(self, examples, drugs):
        feedback_str = ""
        for example in examples:
            for drug in drugs.keys():
                feedback_str += f"{drug} Use: {bool(example[drug])}\n"
        return feedback_str

    def _generate_explanations_and_select_options(self, drugs, explain):
        explanation_and_select_options_str = ""
        for drug in drugs.keys():
            if explain:
                explanation_and_select_options_str += f"Explanation for {drug} Use:\n {gen(name=f'{drug}_explanation', stop=[n,'.'])}\n"
            explanation_and_select_options_str += f"{drug} Use: {select(options=['True', 'False'], name=drug)}\n"
        return explanation_and_select_options_str

if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=0 python -m drugdetector.detect

    detector = DrugDetector(
        model_id="TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ",
        cache_dir="/data2/.shared_models/",
    )

    result = detector.detect(medical_text="Patient denies using heroin but reports cocaine use.")
    print(result)

    drugs = {    
        "Alcohol": "Alcohol is a legal substance but can be abused and lead to addiction and various health issues.",
        "Fentanyl": "Fentanyl is a potent synthetic opioid that is highly addictive and can lead to overdose, especially when used illicitly."
    }

    result = detector.detect(medical_text="Patient was using ethanol and fentanyl for several months.", drugs=drugs, explain=True)
    print(result)
