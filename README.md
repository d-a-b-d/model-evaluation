# Evaluating Diagnostic Recognition Blind Spots in Small Models: A Cross-Model Study of Endometriosis

*Note: this is the submission for the Fatima Fellowship Application Fall 2026 technical challenge.*

*Answer to [Question 1](https://huggingface.co/datasets/duaa-amer/ai-blind-spot-evaluation-dataset#blind-spot-question-1) and [Question 3](https://huggingface.co/datasets/duaa-amer/ai-blind-spot-evaluation-dataset#path-forward-question-3)*

Healthcare practitioners, when faced with atypical manifestations of diseases, often resort to hypothesis generation and revision if classic pattern recognition doesn’t suffice. This raises the question that sits at the heart of this evaluation: **How does a model perform when faced with a presentation of a disease that departs from the classic textbook pattern?**

Standard benchmarks can measure whether a model accurately recognizes a disease. However, aggregate accuracy doesn’t reveal any behavioral information about the robustness of atypical diagnostic recognition.

This is critical in the case of underdiagnosed diseases that manifest through atypical presentations, such as endometriosis. Endometriosis is a disease that can manifest either typically or atypically, with an average diagnostic delay of 7 years [1]. Typical presentations of the disease include pelvic and reproductive symptoms, while atypical presentations include gastrointestinal, urinary, thoracic, and extrapelvic symptoms.

## Blind Spot (Question 1)

I evaluate the ability of a model to accurately recognize endometriosis across both typical *and* atypical manifestations. I do this by asking three questions:

1. Can the model recognize endometriosis across various presentations?
2. Does the addition of certain symptoms provoke a possible symptom bias?
4. Does the model possess explicit knowledge about atypical endometriosis?
   
## Model Choice: MedGemma-1.5-4B-IT vs Qwen 3.5-4B

MedGemma-1.5-4B-IT is a vision-language model developed by Google that is trained for medical applications. It falls within the required 0.6-6B parameter range. It was chosen for this task due to the medical subject of this evaluation.

To investigate the influence of a medically specialized AI like MedGemma-1.5-4B-IT vs a general AI, another general-purpose model had to be chosen. Thus this became a comparative cross-model evaluation.

For this experiment, Qwen 3.5-4B was chosen. Qwen 3.5-4B is a general-purpose model that sits within the required 0.6-6B parameter range.

Both the Qwen and MedGemma models have 4B parameters, and both models were evaluated using the same prompts. Both models were also loaded using 4-bit quantization with float16 computation due to the memory constraints of Google Colab T4 GPU. Greedy decoding was used (do_sample=False) to minimize sampling randomness. In Qwen-3.5-4B's case, thinking mode was disabled (enable_thinking=False).

### Scoring Criteria

For Experiment 1 and 2, the scoring criteria rely on a binary configuration. An answer is considered correct when the model recognizes endometriosis based on the clinical vignette. An answer is considered wrong when it selects any other disease.

For Experiment 3, a model’s answer was considered successful if it explicitly recognized an association between atypical symptoms (organ-specific, extrapelvic, urinary, thoracic) and endometriosis.

## Experiment 1: Presentation Type

There are 12 clinical vignettes divided into three groups: reproductive and pelvic, organ-specific, and extrapelvic presentations. The first is considered a typical manifestation of the disease, while the latter two are considered atypical. Each group has four cases.


| **Presentation** | **MedGemma 1.5-4B-IT** | **Qwen3.5-4B** |
| ----------------------- | ------------------- | -------------- |
| **Pelvic/reproductive (typical)** | **4/4** | **4/4** |
| **Organ-specific (atypical)** | **0/4** | **2/4** |
| **Extrapelvic (atypical)** | **0/4** | **1/4** |

MedGemma recognized endometriosis as the diagonsis in all of the featured pelvic/reproductive cases. However, it did not recognize any of the atypical organ-specific and extrapelvic presentations.

Like MedGemma, the Qwen model recognized all classic cases of endometriosis. However, during atypical recognition of endometriosis, it recognized two of the organ-specific cases and one of the extrapelvic cases.

Since MedGemma recognized only pelvic/reproductive manifestations of endometriosis, one possibility was symptom bias. 

## Experiment 2: Single Symptom Bias

I therefore tested the stability of diagnostic recognition when adding one symptom to a baseline endometriosis presentation. 

There are 12 clinical vignettes based on three baselines. For each baseline, gastrointestinal, urinary, and unrelated symptoms are separately added. 


| **Condition** | **MedGemma 1.5-4B-IT** | **Qwen3.5-4B** |
| --------------------- | ------------------- | -------------- |
| **Baseline** | **3/3** | **3/3** |
| **Gastrointestinal** | **0/3** | **3/3** |
| **Urinary** | **1/3** | **3/3** |
| **Unrelated Symptom** | **2/3** | **3/3** |

During the MedGemma evaluation, adding recurrent diarrhea shifted all three diagnoses away from endometriosis. Adding urinary frequency shifted two of three diagnoses. Finally, adding an unrelated tension headache shifted only one diagnosis.

However, Qwen successfully recognized all cases as endometriosis.

## Experiment 3: Knowledge Probe

Additionally, both models' explicit knowledge of gastrointestinal, urinary, extrapelvic, and thoracic endometriosis was tested.

| **Knowledge** | **MedGemma 1.5-4B-IT** | **Qwen3.5-4B** |
| -------------------- | ------------------- | -------------- |
| **Gastrointestinal** | **Pass** | **Pass** |
| **Urinary** | **Pass** | **Pass** |
| **Extrapelvic** | **Pass** | **Pass** |
| **Thoracic** | **Pass** | **Pass** |

Both models demonstrated explicit knowledge of atypical endometriosis. However, this knowledge was not consistently reflected in their diagnostic results.

## Cross-Model Results

Both models performed equally on recognition of typical endometriosis presentation. However, Qwen had better results when recognizing atypical presentations. It also demonstrated better performance when tested for stability under controlled symptom addition.

Overall, the results suggest that model domain specialization doesn’t always guarantee better performance in diagnostic application.

## Path Forward (Question 3)

**First, I look towards data curation strategies.** Here, the challenge is twofold:
- *Diversity of clinical presentations.* Therefore I  propose the inclusion of more atypical cases in the training data, especially cases where endometriosis occurs alongside unrelated symptoms. 
- *Data ambiguity.* Endometriosis has an average diagnostic delay of approximately 7 years. In the case of clinical ambiguity, data curation becomes more difficult. As a way forward, expert clinicians should review and annotate the training data. Clinicians should also distinguish between confirmed endometriosis, suspected endometriosis, and other overlapping symptoms/conditions in the data. 

**Second, I suggest fine-tuning strategies.** Proposed strategies include:
- *Contrastive fine-tuning.* Here, pairs of similar presentations could help models distinguish endometriosis from overlapping conditions.
- *Targeted supervised fine-tuning.* Fine-tuning on cases that cover atypical manifestations, and not just typical manifestations, might improve model output.

## Limitations

- In Experiment 1, the cases provided differed in several clinical aspects. Experiment 2 attempts to address this by keeping the baseline case constant and changing only one symptom.

- This evaluation uses a small number of synthetic clinical vignettes sourced from medical guidelines and case reports; thus the results may not be generalizable to other presentations/models.

- The model’s observable output isn’t representative of its internal computation. This limits any serious comparisons to human hypothesis generation and revision in the case of atypical presentations.

- MedGemma-1.5-4B-IT exhibits prompt sensitivity; therefore prompt construction affects its output.

## Artifacts

This bucket contains the following artifacts: 

- README.md
- evaluation-dataset.csv
- medgemma_1_5_4b_it_evaluation.ipynb
- qwen3_5_4b_evaluation.ipynb
- clinical_vignette_sources.csv (Note: contains the sources and medical literature used to develop the clinical vignettes).


## References

1. Breton Z, Gouesbet S, Indersie E, Gabillet M, Tran VT, Aflak N, Borghese B, Petit E, Roman H, Millepied AC, Nève De Mevergnies M, Kvaskoff M. Endometriosis Diagnostic Delay and Its Correlates: Results from the ComPaRe-Endometriosis Cohort. J Womens Health (Larchmt). 2026 Jan;35(2):172-188. doi: 10.1177/15409996251380129. Epub 2026 Feb 2. PMID: 40999898.
