# TraumaICDBERT: A Natural Language Processing Algorithm to Automate Injury ICD-10 Diagnosis Code Extraction from Free Text 

A BERT-based deep learning NLP algorithm to extract ICD-10 codes from unstructured tertiary survey notes of trauma patients.


<img src="https://github.com/asivura/trauma-icd/blob/main/figures/Trauma%20ICD%20Overview.png" width="500">


# Usage

1. Manually collect training data (ICD10 codes) from each EMR. See example here: [Google sheet](https://docs.google.com/spreadsheets/d/19PKbWvzFohSQhzaMaz9lvfDuOqMZI8ZJM7aqzZ57Xeg/edit?usp=sharing)

(Steps 2-6 are included in the Demo Notebook `main-demo.ipynb`, so there is no need to run the notebooks below individually if you chose to use `main-demo.ipynb`.)

2. Obtain a list of ICD-10 candidate codes from the official schema `notebooks/ohdsi-vocab.ipynb`
3. Create a list of training, test, and validation patient_ids using `notebooks/prepare-dataset.ipynb`
4. Prepare the input-label dataset using `notebooks/prepare-dataset.ipynb`
5. Generate ICD-10 code semantic similarity scores using `notebooks/gpt3-embeddings.ipynb`
6. Run the training script using a Google Colab notebook: `notebooks/train-colab.ipynb`, or using command line:

7. The evaluation scripts for Amazon Web Services Comprehend Medical is available in the `/notebooks` directory.


# Results 

|               | Top 10 codes | Top 10 codes | Top 50 codes | Top 50 codes| All 170 codes | All 170 codes|
|---------------|:------------------------------------------------:|:-------------------------:|:------------------------------------------------:|:-------------------------:|:--------------------------------------:|:-------------------------:|
|               |                  TraumaICD- BERT                 | Amazon Comprehend Medical |                  TraumaICD- BERT                 | Amazon Comprehend Medical |             TraumaICD- BERT            | Amazon Comprehend Medical |
|   AUROC_micro  |                       98.6                       |            82.6           |                       96.0                       |            80.5           |                  95.7                  |            80.0           |
|   AUROC_macro  |                       98.1                       |            79.1           |                       92.8                       |            76.2           |                  90.0                  |            70.9           |
| AUROC_weighted |                       98.1                       |            82.4           |                       93.3                       |            79.4           |                  92.2                  |            77.4           |
|    F1_micro    |                       87.0                       |            59.6           |                       72.6                       |            43.4           |                  66.4                  |            32.2           |
|    F1_macro    |                       84.1                       |            53.8           |                       66.5                       |            36.8           |                  41.8                  |            18.8           |
|   F1_weighted  |                       87.1                       |            61.4           |                       72.0                       |            47.4           |                  65.6                  |            41.0           |
|  Precision@5  |                       36.6                       |            30.2           |                       46.6                       |            31.2           |                  47.7                  |            26.6           |
|  Precision@10 |                       18.1                       |            18.1           |                       27.9                       |            20.8           |                  29.3                  |            19.3           |
|  Precision@20 |                       18.1                       |            18.1           |                       15.2                       |            12.1           |                  16.3                  |            11.6           |
|    Recall@5   |                       99.4                       |            85.7           |                       83.4                       |            59.1           |                  75.2                  |            46.3           |
|   Recall@10   |                       100.0                      |           100.0           |                       93.0                       |            71.0           |                  85.1                  |            60.5           |
|   Recall@20   |                       100.0                      |           100.0           |                       98.4                       |            79.5           |                  91.1                  |            67.1           |

Comparative performance of TraumaICDBERT and Amazon Web Service Comprehend Medical for predicting 4-character injury ICD-10 diagnosis codes from free text (tertiary survey notes). TraumaICDBERT's performance exceeded or matched that of Amazon Web Service Comprehend Medical across all metrics.

# Training Details

The training task is formatted as follows: the input consists of one ICD-10 code definition (e.g., Multiple fractures of ribs) and the tertiary EMR (imaging report + tertiary impression), the output is a probability between 1 (positive ground truth example) and 0 (negative example). For each patient, we generate 10/50/170 such input-output pairs, where the label for positive example is 1, and negative example is 0. We fine-tune the model using a pre-trained BioLinkBERT on this classification task. 

During training, the model is trained on all the positive examples, as well as an equal number of negative examples. Since there are more negative examples than positive examples (a patient may have 4 injuries while there are 170 candidate codes), we randomly sample a subset of the negative examples at each epoch. 

During reduced_validation, the model is evaluated on all the positive examples and a subset of the negative examples (to improve computational efficiency, we select only the most challenging codes that have similar GPT-3 embeddings with the positive code). 

When the model is finished training (e.g., for 20 epochs), the model is evaluated in a final loop of all positive examples and all negative examples. For example, each EMR will be inferenced 170 times, where each inference corresponds to a candidate ICD-10 code. The model evaluation metrics is computed using this probability matrix.

The training time of BioLinkBERT-base for 20 epochs on a P100 GPU is less than a day.

# Hardware Requirements

The entire tokenized dataset (encoded_ds) must be able to fit onto the computer RAM. We used a workstation with 24GB of RAM when using a dataset of 3,500 EMRs, where each EMR has 170 input-output pair examples.

The recommended batch size for training BioLinkBERT-base on P100 GPU (16GB VRAM) is 16 for training and 32 for evaluation.

The recommended batch size for training BioLinkBERT-large on P100 GPU (16GB VRAM) is 6 for training and 12 for evaluation.


# Support
If you have any questions regarding this repository, please open a support ticket on Github, or write to yifuchen [a] stanford [.] edu for guidance.
