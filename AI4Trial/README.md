# AI4Trial: AI-Ready Clinical Trial Datasets


## 🚀 Installation 

You can install the remaining dependencies for our package by executing:
```
pip install -r requirements.txt
```
Please note, our package has been tested and confirmed to work with Python 3.7. We recommend using this version to ensure compatibility and optimal performance.

## Download

Download the relevant supporting documents in the [link](https://zjueducn-my.sharepoint.com/:f:/g/personal/yaojunhu_zju_edu_cn/EsgIPd3QyjVMipw6tkPv3hoBRk83HqO4X_laZhf6nD87IA?e=0O8fFi) and put them in ```data/```.

## Trialbench

This reposity load Trialbench from [Huggingface](https://huggingface.co/datasets/ML2Healthcare/ClinicalTrialDataset). To quickly understand the data, we also provide some toy samples in [Trialbench]()


## 💻 Usage 
To run mortality rate prediction task, you can use the following code.
```
cd AI4Trial
python learn_multi_model.py --base_name mortality_rate --phase 'Phase 1' --exp Temp
```

## 💼 Support
If you need help with the tool, you can raise an issue on our GitHub issue tracker. For other questions, please contact our team. 



