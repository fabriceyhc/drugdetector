import dspy

DOC_STRING = """
Determine whether the medical text contains reference to illicit drug use.
Specifically look for mentions of the following drugs:
    Heroin: Heroin is an illegal opioid drug known for its high potential for addiction and overdose.
    Cocaine: Cocaine is a powerful stimulant drug that is often abused for its euphoric effects.
    Methamphetamine: Methamphetamine (including illicit amphetamine use, but not prescribed amphetamines for ADHD) is a potent central nervous system stimulant that is highly addictive.
    Benzodiazepine: Benzodiazepines are a class of psychoactive drugs commonly prescribed for anxiety, insomnia, and other conditions but can be abused for their sedative effects.
    Prescription Opioids: Prescription opioids (only if being misused or used illicitly, not if taken as prescribed) are medications typically prescribed for pain relief but can be highly addictive when misused.
    Cannabis: Cannabis, also known as marijuana, is often used recreationally or medicinally but can be illegal depending on the jurisdiction.
    Injection Drugs: Injection drug use (IDU, IVDA, IVDU) refers to the use of drugs administered via needles, often associated with higher risks of infectious diseases.
    General Drugs: General drug use refers to the use of any illegal or illicit substances.
Special Notes:
    1. The mere mention of a drug is not sufficient. You are only looking for illicit use of the drug in the medical note. Do not assume that a drug is being used illicitly without some evidence. 
    2. If the text warns against the use of a particular drug, that does not mean the patient is actually using the drug. 
    3. If family drug use is present, that is not relevant to the patient and should not be flagged. 
    4. If the patient denies using a particular drug, do not mark that drug as being present. For example, if the note says "patient denied using heroin", then the label should be False. 
    5. Many opioids and benzodiazepines are appropriately used and should not be noted. We only want you to identify cases where the patient is not using them appropriately. For example, if they are taking Percocets acquired from friends or from the streets, this would be considered illicit misuse. 
    6. Medical recommendations about drugs do not mean the patient is actually using the drug.          
"""
MEDICAL_TEXT = "A medical text containing potentially illicit drug use by a patient."
HEROIN = "Whether the illicit use of heroin is detected in the medical note."
COCAINE = "Whether the illicit use of cocaine is detected in the medical note."
METHAMPHETAMINE = "Whether the illicit use of methamphetamines is detected in the medical note."
BENZODIAZEPINE = "Whether the illicit use of benzodiazepines is detected in the medical note."
RX_OPIOID_MISUSE = "Whether the illicit use of prescription opioids is detected in the medical note."
CANNABIS = "Whether the illicit use of cannabis is detected in the medical note."
IDVU = "Whether the illicit use of injection drugs is detected in the medical note."
GENERAL = "Whether the illicit use of any drug is detected in the medical note."
EXPLANATION = "Explanation for the detection status."

class MultilabelEL(dspy.Signature):
    DOC_STRING
    medical_text: str = dspy.InputField(desc=MEDICAL_TEXT)
    explanation: str = dspy.OutputField(desc=EXPLANATION)
    heroin: bool = dspy.OutputField(desc=HEROIN)
    cocaine: bool = dspy.OutputField(desc=COCAINE)
    methamphetamine: bool = dspy.OutputField(desc=METHAMPHETAMINE)
    benzodiazepine: bool = dspy.OutputField(desc=BENZODIAZEPINE)
    rx_opioid_misuse: bool = dspy.OutputField(desc=RX_OPIOID_MISUSE)
    cannabis: bool = dspy.OutputField(desc=CANNABIS)
    injection_drug_use: bool = dspy.OutputField(desc=IDVU)
    general_drug_use: bool = dspy.OutputField(desc=GENERAL)

class MultilabelL(dspy.Signature):
    DOC_STRING
    medical_text: str = dspy.InputField(desc=MEDICAL_TEXT)
    heroin: bool = dspy.OutputField(desc=HEROIN)
    cocaine: bool = dspy.OutputField(desc=COCAINE)
    methamphetamine: bool = dspy.OutputField(desc=METHAMPHETAMINE)
    benzodiazepine: bool = dspy.OutputField(desc=BENZODIAZEPINE)
    rx_opioid_misuse: bool = dspy.OutputField(desc=RX_OPIOID_MISUSE)
    cannabis: bool = dspy.OutputField(desc=CANNABIS)
    injection_drug_use: bool = dspy.OutputField(desc=IDVU)
    general_drug_use: bool = dspy.OutputField(desc=GENERAL)