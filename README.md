# GRACE

#### This is the official repository of source code for the [paper](https://arxiv.org/abs/1911.02042):

GRACE: Generating Concise and Informative Contrastive Sample to Explain Neural Network Model’s Prediction. Thai Le, Suhang Wang, Dongwon Lee. 26th ACM SIGKDD Int’l Conf. on Knowledge Discovery and Data Mining (KDD), Virtual. August 2020.

#### (October 26) We updated the arxiv version of our paper to reflect the handling of categorical features of GRACE algorithm.
#### (December 13) GRACE with black-box attack code is available at https://github.com/research0610/MOCHI (Thanks to the author)

#### Train, Evaluation and Explanation
Use `main.py` file for training, evaluating and generating explanation.

Example on Spam Detection Dataset:
```
python main.py --csv spam.csv --hiddens 50 30 --lr 0.001 --gen_gamma 0.5 --gen_max_features 5 --explain_units %
```

Outputs:
```
Training...
Val loss: 0.4463 Val acc: 0.8213
Val loss: 0.1990 Val acc: 0.9324
Val loss: 0.1871 Val acc: 0.9300

  Dataset     Accuracy    F1
=============================
Validation    0.930     0.930
Test          0.933     0.933
Generating Contrastive Sample...100%

 Dataset   #avgFeatChanged   Fidelity
====================================
Test           1.254         1.000

         sample  prediction word_freq_make word_freq_address word_freq_all char_freq_%24
0     Original           0          0.000             0.000         1.200         0.000
1  Contrastive           1          0.000             0.000         1.200         1.153
EXPLANATION:  "IF char_freq_%24 increased 1.153 %, the model would have predicted 1 RATHER THAN 0"
```

Other Example Datasets:
```
python main.py --csv eegeye.csv --hiddens 40 30 --lr 0.01 --gen_gamma 0.5 --gen_max_features 5
python main.py --csv phoneme.csv --hiddens 20 5 --lr 0.001 --gen_gamma 0.5 --gen_max_features 5
python main.py --csv tokyo1.csv --hiddens 50 20 --lr 0.01 --gen_gamma 0.5 --pre_scaler 1 --gen_max_features 5
python main.py --csv mfeat.csv --hiddens 100 50 --lr 0.001 --gen_gamma 0.5 --pre_scaler 1 --gen_max_features 5
python main.py --csv diabetes.csv --hiddens 10 10 --lr 0.001 --gen_gamma 0.5 --gen_max_features 5
python main.py --csv vehicle.csv --hiddens 30 10 --lr 0.001 --gen_gamma 0.5 --gen_max_features 5
```

#### Citation
```
@article{le2019grace,
    title={GRACE: Generating Concise and Informative Contrastive Sample to Explain Neural Network Model's Prediction},
    author={Thai Le and Suhang Wang and Dongwon Lee},
    year={2019},
    journal={Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '20)},
    doi={10.1145/3394486.3403066}
    isbn={978-1-4503-7998-4/20/08}
}
```
