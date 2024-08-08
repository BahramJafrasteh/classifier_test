<style>
body {
    background-color: black; /* Set background to black */
    color: white; /* Set text color to white for visibility */
}

td, th {
    border: none !important;
    color: white; /* Ensure table text is also visible */
}

code {
    background-color: #333; /* Dark background for code blocks */
    color: #0f0; /* Green text for code for contrast */
    padding: 2px 4px; /* Add some padding for better readability */
    border-radius: 4px; /* Rounded corners for code blocks */
}
</style>

# "Is Your Classifier Better than Mine? Pitfalls in Statistical Tests for Model Comparison by Cross-Validation"

This document explains how p-values can be influenced by various factors, including the number of cross-validations (CV), repetitions, and sample sizes.

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
   git clone classifier_test
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




