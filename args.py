from argparse import ArgumentParser
parser = ArgumentParser(description='GRACE: Generating Concise and Informative Contrastive Sample')
parser.add_argument('--csv', type=str, default="spam.csv")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_reduce_rate', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--hiddens', nargs='+', type=int, default=[50, 30])
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_epochs', type=int, default=500)
parser.add_argument('--pre_scaler', type=int, default=0)
parser.add_argument('--model_scaler', type=int, default=1)

parser.add_argument('--gen_max_features', type=int, default=5)
parser.add_argument('--gen_gamma', type=float, default=0.5)
parser.add_argument('--gen_overshoot', type=int, default=0.0001)
parser.add_argument('--gen_max_iter', type=int, default=50)

parser.add_argument('--num_normal_feat', type=int, default=3)
parser.add_argument('--explain_table', type=int, default=1)
parser.add_argument('--explain_text', type=int, default=1)

parser.add_argument('--verbose_threshold', type=int, default=50)
parser.add_argument('--model_temp_path', type=str, default="./model_temp.pt")


parser.add_argument('--seed', type=int, default=77)
args = parser.parse_args()