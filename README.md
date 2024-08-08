# "Is Your Classifier Better than Mine? Pitfalls in Statistical Tests for Model Comparison by Cross-Validation"


The manuscript explains how p-values can be influenced by various factors, including the number of cross-validations (CV), repetitions, and sample sizes.
The code to reproduce the results is here.
---

## Table of Contents
- [Dependencies](#dependencies)
- [How to Use](#how-to-use)
- [Citation and Acknowledgements](#citation-and-acknowledgements)

## Dependencies
This software requires the following libraries:
```
torch
numpy
pickle
sklearn
MLP_model
pandas
```


## How to Use
Installing the is straightforward. Follow these steps:

1. **Clone the Library**:
   ```bash
   git clone https://github.com/BahramJafrasteh/classifier_test.git
   cd classifier_test
2. **Install the required packages**
3. **Run the Classifier**:
    ```bash
    python main_classifier --dataset_name <dataset> --model_name <model_type>
Replace <dataset> with the name of the dataset you wish to process (adni, abide, or abcd5 for ADNI, ABIDE, and ABCD datasets, respectively).
Replace <model_type> with the desired model type (e.g., MLP or logistic regression).
The results will be saved in a `.pkl` file, which can be used for visualizing and creating plots.


 -------- 

#Citation and acknowledgements


```
This work is currently under review.

```




