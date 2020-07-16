# GRACE

### This is the official repository of source code for the [paper](http://pike.psu.edu/publications/kdd20-grace.pdf):

### GRACE: Generating Concise and Informative Contrastive Sample to Explain Neural Network Model’s Prediction. Thai Le, Suhang Wang, Dongwon Lee. 26th ACM SIGKDD Int’l Conf. on Knowledge Discovery and Data Mining (KDD), Virtual. August 2020.
(Being Updated)

### Train, Evaluation and Explanation

python main.py --csv eegeye.csv --hiddens 40 30 --lr 0.01 --gen_gamma 0.5 --gen_max_features 5
python main.py --csv spam.csv --hiddens 50 30 --lr 0.001 --gen_gamma 0.5 --gen_max_features 5
python main.py --csv phoneme.csv --hiddens 20 5 --lr 0.001 --gen_gamma 0.5 --gen_max_features 5
python main.py --csv tokyo1.csv --hiddens 50 20 --lr 0.01 --gen_gamma 0.5 --pre_scaler 1 --gen_max_features 5
python main.py --csv mfeat.csv --hiddens 100 50 --lr 0.001 --gen_gamma 0.5 --pre_scaler 1 --gen_max_features 5
python main.py --csv diabetes.csv --hiddens 10 10 --lr 0.001 --gen_gamma 0.5 --gen_max_features 5
python main.py --csv vehicle.csv --hiddens 30 10 --lr 0.001 --gen_gamma 0.5 --gen_max_features 5


### Citation
@article{le2019grace,
    title={GRACE: Generating Concise and Informative Contrastive Sample to Explain Neural Network Model's Prediction},
    author={Thai Le and Suhang Wang and Dongwon Lee},
    year={2019},
    journal={Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '20)},
    doi={10.1145/3394486.3403066}
	isbn={978-1-4503-7998-4/20/08}
}