# drugdetector

This project provides a simple wrapper for performant zero-shot drug detection using LLMs. 

## Install

```
pip install drugdetector
```

Optionally, but recommended is to install CUDA support for `llama-cpp-python` ([source](https://github.com/abetlen/llama-cpp-python)).

```
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda-version>
```

Where `<cuda-version>` is one of the following:

cu121: CUDA 12.1
cu122: CUDA 12.2
cu123: CUDA 12.3
cu124: CUDA 12.4
cu125: CUDA 12.5

You can find what version of CUDA you support via `nvidia-smi` and looking for the `CUDA Version` in the top right corner. 

## Demo

For this demo, we're using our fine-tuned 8-bit quantized Llama-3 model (`fabriceyhc/Llama-3-8B-DrugDetector`), but you can use many LLMs available on huggingface. 
We recommend one of the following fine-tuned models for improved performance:

- `fabriceyhc/Llama-3-8B-DrugDetector`
- `fabriceyhc/Llama-3-70B-DrugDetector`

```
from drugdetector import DrugDetector

detector = DrugDetector(model_id="fabriceyhc/Llama-3-8B-DrugDetector")
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