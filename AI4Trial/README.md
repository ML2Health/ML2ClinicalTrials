# TrialBench: AI-Ready Clinical Trial Datasets


<p align="center"><img src="./trial.pdf" alt="logo" width="810px" /></p>

This repository contains code for training and testing benchmark models on Trialbench datasets. The provided scripts facilitate the evaluation of various machine learning algorithms, enabling researchers to assess their performance on different clinical trial phases and tasks.

## 🚀 Installation 
We recommend creating a dedicated virtual environment (such as conda) with Python 3.7+ to ensure consistent performance. Once your environment is ready, install the required dependencies:
```
pip install -r requirements.txt
```

## 🔩 Download
All necessary supporting documents can be downloaded from this [link](https://zjueducn-my.sharepoint.com/:f:/g/personal/yaojunhu_zju_edu_cn/EsgIPd3QyjVMipw6tkPv3hoBRk83HqO4X_laZhf6nD87IA?e=0O8fFi). Place them into the `data/` folder.

## 📚 Trialbench
This repository automatically fetches the Trialbench dataset from [Huggingface](https://huggingface.co/datasets/ML2Healthcare/ClinicalTrialDataset). For a quick understanding or experimentation, we also provide toy samples in [Trialbench]().

## 💻 Usage
To run a mortality rate prediction experiment, navigate into the `AI4Trial` directory and use:
```
cd AI4Trial
python learn_multi_model.py --base_name mortality_rate --phase 'Phase 1' --exp Temp
```
Feel free to explore other tasks by adjusting the parameters accordingly.

## 💼 Support
If you encounter any issues or have questions, please open an issue on GitHub. For additional help, reach out to our team. 



