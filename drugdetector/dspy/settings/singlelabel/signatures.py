import dspy

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

class SinglelabelEL(dspy.Signature):
    DOC_STRING
    medical_text: str = dspy.InputField(desc=MEDICAL_TEXT)
    drug_name: str = dspy.InputField(desc=DRUG_NAME)
    drug_description: str = dspy.InputField(desc=DRUG_DESC)
    explanation: str = dspy.OutputField(desc=EXPLANATION)
    label: bool = dspy.OutputField(desc=LABEL)

class SinglelabelL(dspy.Signature):
    DOC_STRING
    medical_text: str = dspy.InputField(desc=MEDICAL_TEXT)
    drug_name: str = dspy.InputField(desc=DRUG_NAME)
    drug_description: str = dspy.InputField(desc=DRUG_DESC)
    label: bool = dspy.OutputField(desc=LABEL)