import warnings
warnings.filterwarnings("ignore")

import unittest
from drugdetector.detect import DrugDetector

class TestDrugDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = DrugDetector(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            cache_dir="/data2/.shared_models/",
            drugs=None
        )

    def test_0_default_drugs(self):
        default_drugs = {
            "Heroin": "Heroin is an illegal opioid drug known for its high potential for addiction and overdose.",
            "Cocaine": "Cocaine is a powerful stimulant drug that is often abused for its euphoric effects.",
            "Methamphetamine": "Methamphetamine (including illicit amphetamine use, but not prescribed amphetamines for ADHD) is a potent central nervous system stimulant that is highly addictive.",
            "Benzodiazepine": "Benzodiazepines are a class of psychoactive drugs commonly prescribed for anxiety, insomnia, and other conditions but can be abused for their sedative effects.",
            "Prescription Opioids": "Prescription opioids (only if being misused or used illicitly, not if taken as prescribed) are medications typically prescribed for pain relief but can be highly addictive when misused.",
            "Cannabis": "Cannabis, also known as marijuana, is often used recreationally or medicinally but can be illegal depending on the jurisdiction.",
            "Injection Drugs": "Injection drug use (IDU, IVDA, IVDU) refers to the use of drugs administered via needles, often associated with higher risks of infectious diseases.",
            "General Drugs": "General drug use refers to the use of any illegal or illicit substances.",
        }
        medical_text = "Patient denies using heroin but reports cocaine use."
        result = self.detector.detect(medical_text)
        print(result)

        # decision for all drugs are present
        self.assertTrue(set(default_drugs).issubset(result))

        # check specific answers that should be True
        self.assertTrue(result["Cocaine"])
        self.assertTrue(result["General Drugs"])

        # check specific answers that should be False
        self.assertFalse(result["Heroin"])
        self.assertFalse(result["Methamphetamine"])
        self.assertFalse(result["Benzodiazepine"])
        self.assertFalse(result["Prescription Opioids"])
        self.assertFalse(result["Cannabis"])
        self.assertFalse(result["Injection Drugs"])


    def test_1_custom_drugs(self):
        new_drugs = {    
            "Alcohol": "Alcohol is a legal substance but can be abused and lead to addiction and various health issues.",
            "Fentanyl": "Fentanyl is a potent synthetic opioid that is highly addictive and can lead to overdose, especially when used illicitly."
        }
        medical_text = "Patient heavily used ethanol for several mos, but denies any fentanyl use."
        result = self.detector.detect(medical_text, drugs=new_drugs)
        print(result)

        # decision for all drugs are present
        self.assertTrue(set(new_drugs).issubset(result))

        # check specific answers that should be True
        self.assertTrue(result["Alcohol"])

        # check specific answers that should be False
        self.assertFalse(result["Fentanyl"])


    def test_2_custom_drugs_w_explanations(self):
        new_drugs = {    
            "Alcohol": "Alcohol is a legal substance but can be abused and lead to addiction and various health issues.",
            "Fentanyl": "Fentanyl is a potent synthetic opioid that is highly addictive and can lead to overdose, especially when used illicitly."
        }
        medical_text = "Patient heavily used ethanol for several mos, but denies any fentanyl use."
        result = self.detector.detect(medical_text, drugs=new_drugs, explain=True)
        print(result)

        # decision for all drugs are present
        self.assertTrue(set(new_drugs).issubset(result))

        # explanations for all drugs are present
        explanation_keys = [f"{drug}_explanation" for drug in new_drugs.keys()]
        self.assertTrue(set(explanation_keys).issubset(result))

        # check specific answers that should be True
        self.assertTrue(result["Alcohol"])

        # check specific answers that should be False
        self.assertFalse(result["Fentanyl"])

if __name__ == '__main__':
    unittest.main()