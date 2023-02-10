import argparse


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("-lr", dest="LR", type=float, default=1e-3)
    parser.add_argument("-epochs", dest="EPOCHS", type=int, default=1000,
                        help="Number of epochs for offline training")
    parser.add_argument("-l2reg", dest="weight_decay", type=float, default=0)
    parser.add_argument("-critic_iterations", dest="critic_iterations", type=int, default=5)
    parser.add_argument("-gradient_penalty", dest="GP_hyperparam", type=float, default=10.)
    parser.add_argument("-WAEreg", dest="WAE_regularization_term", type=float, default=1.)

    parser.add_argument("-dropout", dest="DROPOUT", type=float, default=0.2)
    parser.add_argument("-embedding", dest="EMBEDDING", type=int, default=4)
    parser.add_argument("-hidden", dest="HIDDEN_DIMS", type=int, action="append")
    parser.add_argument("-n_layers", dest="LSTM_LAYERS", type=int, default=2)
    parser.add_argument("-batch_size", dest="BATCH_SIZE", type=int, default=200)

    parser.add_argument("-sw", dest="sparsity_weight", type=float, default=1.,
                        help="Sparsity weight for Sparse AE")
    parser.add_argument("-sp", dest="sparsity_parameter", type=float, default=0.05,
                        help="Sparsity parameter for Sparse AE")

    parser.add_argument("-att_heads", dest="NHEADS", type=int, default=8,
                        help="Number of attention heads")

    parser.add_argument("-feats", dest="FEATS", choices=["analog", "digital", "all"], default="analog",
                        help="Which sensors to use")

    parser.add_argument("-SI", dest="successive_iters", type=int, default=10)
    parser.add_argument("-delta_worse", dest="delta_worse", type=float, default=0.02)
    parser.add_argument("-delta_better", dest="delta_better", type=float, default=0.001)

    parser.add_argument("-model", dest="MODEL_NAME", choices=["lstm_ae", "lstm_sae", "multi_enc_sae", "multi_enc_ae",
                                                              "lstm_all_layer_sae", "diff_comp_sae", "diff_comp_ae",
                                                              "GAN", "SimpleDiscriminator", "LSTMDiscriminator",
                                                              "ConvDiscriminator"],
                        required=True)

    parser.add_argument("-recons_error", dest="reconstruction_error_metric", choices=["dtw", "mse"], default="mse")
    parser.add_argument("-dtw_local", dest="dtw_local_size", type=int, default=5)

    parser.add_argument("-separate_comp", dest="separate_comp", action="store_true")
    parser.add_argument("-init", dest="INIT_LOOP", type=int, default=0)
    parser.add_argument("-end", dest="END_LOOP", type=int, default=17)
    parser.add_argument("-force-training", dest="force_training", action="store_true")
    args = parser.parse_args()
    return args
