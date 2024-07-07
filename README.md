# drugdetector

This project provides a simple wrapper for performant zero-shot drug detection using LLMs. 

## Install

```
pip install drugdetector
```

## Demo

For this demo, we're using a 4-bit quantized Llama-3 8B, but you can use many LLMs available on huggingface. 
We recommend one of the following fine-tuned models for improved performance:

- `fabriceyhc/Meta-Llama-3-8B-Instruct-DrugDetection-v3`
- `fabriceyhc/Meta-Llama-3-70B-Instruct-DrugDetection-v3`

```
from drugdetector import DrugDetector

detector = DrugDetector(model_id="TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ")
```


### Detection on default substances
When no specific drugs are requested, default to the ones we've studied.

```
results = detector.detect("Patient denies using heroin but reports cocaine use.")
print(results)
```

```
{'Heroin': False,
 'Cocaine': True,
 'Methamphetamine': False,
 'Benzodiazepine': False,
 'Prescription Opioids': False,
 'Cannabis': False,
 'Injection Drugs': False,
 'General Drugs': True,
 'medical_text': 'Patient denies using heroin but reports cocaine use.',
 'time_taken': 9.533252477645874}
```


### Detection with custom drug dict
Tests zero-shot accuracy on any substances provided in as a python dictionary in the form of:
```
new_substances = {
  "substance1":"description1",
  "substance2":"description2",
  ...
  "substanceN":"descriptionN",
}
```

```
new_substances = {
    "Alcohol": "Alcohol is a legal substance but can be abused and lead to addiction and various health issues.",
    "Fentanyl": "Fentanyl is a potent synthetic opioid that is highly addictive and can lead to overdose, especially when used illicitly."
}

results = detector.detect("Patient hx of ethl use.", drugs=new_substances)
print(results)
```

```
{'Alcohol': True,
 'Fentanyl': False,
 'medical_text': 'Patient hx of ethl use.',
 'time_taken': 2.379117965698242}
```
### Detection with custom drug dict and explanations provided
Demonstrates the model's ability to explain its decisions.
```
new_drugs = {
    "Alcohol": "Alcohol is a legal substance but can be abused and lead to addiction and various health issues.",
    "Fentanyl": "Fentanyl is a potent synthetic opioid that is highly addictive and can lead to overdose, especially when used illicitly."
}

results = detector.detect("Patient hx of ethl use.", drugs=new_substances, explain=True)
print(results)
```

```
{'Alcohol': True,
 'Fentanyl': False,
 'medical_text': 'Patient hx of ethl use.',
 'time_taken': 45.255696296691895,
 'Alcohol_explanation': 'The medical note mentions "ethl use", which is likely referring to ethanol, the scientific name for alcohol.',
 'Fentanyl_explanation': 'There is no mention of Fentanyl use in the medical note.'}
```