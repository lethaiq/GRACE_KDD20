from args import *
from src.explain import *
from src.fcn import *
from src.grace import *
from src.methods import NaiveGradient
from src.selector import *
from src.trainer import *
from src.utils import *


def load_model(train_data, args):
    num_feat = train_data.getX().shape[1]
    num_class = len(np.unique(train_data.gety()))
    scaler = StandardScaler(with_std=True)
    scaler.fit(train_data.getX())
    stds = np.sqrt(scaler.var_)
    if args.model_scaler:
        model = FCN(num_feat, num_class, args.hiddens, scaler.mean_, stds)
    else:
        model = FCN(num_feat, num_class, args.hiddens)
    return model


def train():
    # load data and model
    scaler, le, _, _, features, train_data, val_data, test_data = read_data(
        args.csv, args.seed, scaler=args.pre_scaler)
    model = load_model(train_data, args)

    # train and test the model
    trainer = Trainer(model, lrate=args.lr, lr_reduce_rate=args.lr_reduce_rate)
    trainer.train(train_dataset=train_data,
                  val_dataset=val_data,
                  patience=args.patience,
                  num_epochs=args.max_epochs,
                  batch_size=args.batch_size)

    torch.save(model.state_dict(), args.model_temp_path)

    _, val_acc, val_f1, val_pred = trainer.validate(val_data)
    _, test_acc, test_f1, test_pred = trainer.validate(test_data)
    print_performance(val_acc, val_f1, test_acc, test_f1)


def test():
    # load data and model
    scaler, le, _, _, features, train_data, val_data, test_data = read_data(
        args.csv, args.seed, scaler=args.pre_scaler)
    model = load_model(train_data, args)
    model.load_state_dict(torch.load(args.model_temp_path))
    trainer = Trainer(model)

    # configurations for generating explanation
    num_feat = train_data.getX().shape[1]
    bound_min, bound_max, bound_type = get_constraints(train_data.getX())
    alphas = args.alpha * \
        np.ones(num_feat) if args.alpha > 0 else np.std(train_data.getX(), axis=0)
    feature_selector = FeatureSelector(train_data.getX(), args.gen_gamma) if args.gen_gamma > 0.0 else None

    avg_feat_changed, fidelity = test_grace(model,
                                            trainer,
                                            test_data.getX(),
                                            args,
                                            method="Naive",
                                            scaler=scaler,
                                            bound_min=bound_min,
                                            bound_max=bound_max,
                                            bound_type=bound_type,
                                            alphas=alphas,
                                            feature_selector=feature_selector)

    print_results(avg_feat_changed, fidelity)


def explain():
    # load data and model
    scaler, le, _, _, features, train_data, val_data, test_data = read_data(
        args.csv, args.seed, scaler=args.pre_scaler)
    model = load_model(train_data, args)
    model.load_state_dict(torch.load(args.model_temp_path))
    trainer = Trainer(model)

    # load generation model
    gen_model = NaiveGradient

    # configurations for generating explanation
    num_feat = train_data.getX().shape[1]
    bound_min, bound_max, bound_type = get_constraints(train_data.getX())
    alphas = args.alpha * \
        np.ones(num_feat) if args.alpha > 0 else np.std(train_data.getX(), axis=0)
    feature_selector = FeatureSelector(train_data.getX(), args.gen_gamma) if args.gen_gamma > 0.0 else None

    # generate explanation on a random sample from test set    
    lb_new = lb_org = 0
    while lb_new == lb_org:
        i = np.random.choice(len(test_data.getX())) # select a random sample from test set
        x = test_data.getX()[i:i+1][0]
        x_var = Variable(torch.from_numpy(x.reshape(1,-1))).type(torch.FloatTensor)

        lb_org, lb_new, x_adv, feats_idx = generate(x_var, model, gen_model, args,
                                                    scaler=scaler, 
                                                    trainer=trainer,
                                                    bound_min=bound_min,
                                                    bound_max=bound_max,
                                                    bound_type=bound_type,
                                                    alphas=alphas,
                                                    feature_selector=feature_selector)

    # show explanation
    # print(features[feats_idx])
    if scaler:
        x = scaler.inverse_transform(x.reshape(1, -1))[0]
    if args.explain_table:
        explain_table(x, x_adv, lb_org, lb_new, feats_idx, features, args.num_normal_feat)
    if args.explain_text:
        explain_text(x, x_adv, lb_org, lb_new, feats_idx, features, units=args.explain_units)


if __name__ == "__main__":
    train() # training a FCN model
    test() # test the trained model with a generation method
    explain() # explain the prediction of trained model
