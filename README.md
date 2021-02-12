# Residual Attention Network implemented with tensorflow 2

## Description
Residual Attention Network[[link](https://arxiv.org/abs/1704.06904)]

## structure

```
.
└── utils/           
    ├── __init__.py
    ├── residual_unit.py                     # residual unit layer
    ├── attention_module.py                  # attention module layer
    ├── brunch.py                            # Mask, Trunk brunch layer
    ├── model.py                             # ResAttNet 
    └── preprocessing.py                     # image prepocessing function
├── .gitignore         
├── requirements.txt   
├── config.txt                               # model config
├── Residual Attention Network.ipynb         # Examples of progress 
└── ResAttNet.py                             # model training and save weight py
```

## Usage

```
python ResAttNet.py --model_save=True
```
 
+ --model_save : Whether to save the generated model weight(bool, default=True)  

## Result


## reference
Wang, Fei, et al. "Residual attention network for image classification." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.