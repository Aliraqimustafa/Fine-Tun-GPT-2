# GPT-2 Fine-Tuning Project

Welcome to the GPT-2 fine-tuning project repository! In this project, we fine-tune the GPT-2 transformer model using the SQuAD v2 dataset. The goal is to create a model that can generate human-like text and answer questions based on the input data.

## Project Overview

In this repository, we perform fine-tuning on the GPT-2 model using the SQuAD v2 data. This project provides a step-by-step guide on how to fine-tune the GPT-2 model with your own dataset. The fine-tuned model can be used for various natural language processing tasks, such as text generation and question answering.

## Quick Implementation (Kaggle)

If you're looking for a swift way to implement this project, follow these steps:

1. Visit [Kaggle](https://www.kaggle.com) and create a new code notebook.
2. Use the provided dataset: [SQuAD 2.0 Datasets](https://www.kaggle.com/datasets/studentmustafaai/datasets-squad-20).
3. Copy all the code cells from the `main.py` file in this repository and paste them into your Kaggle notebook.
4. Run the notebook cells to initiate the fine-tuning process on the GPT-2 model.

## Local Implementation

To implement this project on your computer, follow these steps:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/Aliraqimustafa/Fine-Tun-GPT-2/.git
   ```
2. Navigate to the cloned directory:
   ```bash
   cd Fine-Tun-GPT-2
   ```
3. Install the required dependencies by running:
   ```bash
   pip install transformers
   ```
4. Download the SQuAD v2 dataset from [here](https://www.kaggle.com/datasets/studentmustafaai/datasets-squad-20) and place it in a directory of your choice.
5. Open the ```main.py``` file and set the path variable to the directory containing the SQuAD v2 dataset.
6. run python :
   ```bash
   python main.py
   ```
 
## Contributing
Contributions to this project are welcome! If you have suggestions, improvements, or bug fixes, feel free to submit a pull request.
