import os
import numpy as np
import torch as th
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import pickle as pkl
import tqdm
from torch.autograd import Variable
from ArgumentParser import parse_arguments
from LSTM_AAE import Encoder, Decoder, SimpleDiscriminator, LSTMDiscriminator, ConvDiscriminator

####################
#
# Based on the implementation: https://github.com/schelotto/Wasserstein-AutoEncoders
#
####################


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def train_discriminator(optimizer, train_tensor, random_latent_space, args):

    frozen_params(args.encoder)
    frozen_params(args.decoder)
    free_params(args.discriminator)

    real_latent_space = args.encoder(train_tensor)

    stacked_LV = th.repeat_interleave(real_latent_space,
                                      train_tensor.shape[1],
                                      dim=1).reshape(-1,
                                                     train_tensor.shape[1],
                                                     real_latent_space.shape[-1]).to(args.device)

    discriminator_real = args.discriminator(stacked_LV)
    discriminator_random = args.discriminator(random_latent_space)

    loss_real_term = th.log(discriminator_real)
    loss_random_term = th.log(1-discriminator_random)

    loss = args.WAE_regularization_term * -th.mean(loss_real_term + loss_random_term)
    loss.backward()

    nn.utils.clip_grad_norm_(args.discriminator.parameters(), 1)

    optimizer.step()
    return loss.item()


def train_reconstruction(optimizer_encoder, optimizer_decoder, train_tensor, args):

    free_params(args.encoder)
    free_params(args.decoder)
    frozen_params(args.discriminator)

    real_latent_space = args.encoder(train_tensor)
    stacked_LV = th.repeat_interleave(real_latent_space,
                                      train_tensor.shape[1],
                                      dim=1).reshape(-1,
                                                     train_tensor.shape[1],
                                                     real_latent_space.shape[-1]).to(args.device)

    reconstructed_input = args.decoder(stacked_LV)
    discriminator_real_latent = args.discriminator(Variable(stacked_LV))

    reconstruction_loss = F.mse_loss(reconstructed_input, train_tensor)
    discriminator_loss = args.WAE_regularization_term * (th.log(discriminator_real_latent))

    loss = th.mean(reconstruction_loss - discriminator_loss)
    loss.backward()

    nn.utils.clip_grad_norm_(args.encoder.parameters(), 1)
    nn.utils.clip_grad_norm_(args.decoder.parameters(), 1)

    optimizer_encoder.step()
    optimizer_decoder.step()

    return loss.item()


def train_model(train_tensors,
                epochs,
                lr,
                args):

    optimizer_discriminator = optim.Adam(args.discriminator.parameters(), lr=0.1 * lr)
    optimizer_encoder = optim.Adam(args.encoder.parameters(), lr=lr)
    optimizer_decoder = optim.Adam(args.decoder.parameters(), lr=lr)

    loss_over_time = {"discriminator": [], "encoder/decoder": []}

    multivariate_normal = MultivariateNormal(th.zeros(args.EMBEDDING), th.eye(args.EMBEDDING))

    for epoch in range(epochs):

        discriminator_losses = []
        encoder_decoder_losses = []
        with tqdm.tqdm(train_tensors, unit="cycles") as tqdm_epoch:
            for train_tensor in tqdm_epoch:
                tqdm_epoch.set_description(f"Epoch {epoch + 1}")
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                optimizer_discriminator.zero_grad()

                random_latent_space = multivariate_normal.sample(train_tensor.shape[:2]).to(args.device)
                discriminator_losses.append(train_discriminator(optimizer_discriminator,
                                                                train_tensor,
                                                                random_latent_space,
                                                                args))

                encoder_decoder_losses.append(train_reconstruction(optimizer_encoder,
                                                                   optimizer_decoder,
                                                                   train_tensor,
                                                                   args))

        loss_over_time['discriminator'].append(np.mean(discriminator_losses))
        loss_over_time['encoder/decoder'].append(np.mean(encoder_decoder_losses))

        print(f'Epoch {epoch + 1}: discriminator loss {np.mean(discriminator_losses)} encoder/decoder loss {np.mean(encoder_decoder_losses)}')

    return loss_over_time


def predict(args, test_tensors, tqdm_desc):
    reconstruction_errors = []
    critic_scores = []
    with th.no_grad():
        args.encoder.eval()
        args.decoder.eval()
        args.discriminator.eval()
        with tqdm.tqdm(test_tensors, unit="cycles") as tqdm_epoch:
            for test_tensor in tqdm_epoch:
                tqdm_epoch.set_description(tqdm_desc)
                test_tensor = test_tensor.to(args.device)
                reconstruction = args.decoder(args.encoder(test_tensor))
                reconstruction_errors.append(F.mse_loss(reconstruction, test_tensor).item())
                critic_score = th.mean(args.discriminator(test_tensor))
                critic_scores.append(critic_score.item())

    return reconstruction_errors, critic_scores


def offline_train(args):
    print(f"Starting offline training")

    if args.separate_comp:
        with open(f"{args.data_folder}train_tensors_comp0_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
            train_tensors_comp0 = pkl.load(tensor_pkl)
        with open(f"{args.data_folder}train_tensors_comp1_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
            train_tensors_comp1 = pkl.load(tensor_pkl)
        train_tensors = [[train_tensors_comp0[i].to(args.device),
                          train_tensors_comp1[i].to(args.device)] for i in range(len(train_tensors_comp0))]

    else:
        with open(f"{args.data_folder}train_tensors_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
            train_tensors = pkl.load(tensor_pkl)
            train_tensors = [tensor.to(args.device) for i, tensor in enumerate(train_tensors) if i != 467 and i != 585]

    loss_over_time = train_model(train_tensors,
                                 epochs=args.EPOCHS,
                                 lr=args.LR,
                                 args=args)

    with open(args.results_string("offline"), "wb") as loss_file:
        pkl.dump(loss_over_time, loss_file)

    th.save(args.decoder.state_dict(), args.model_saving_string("WAE_decoder"))
    th.save(args.encoder.state_dict(), args.model_saving_string("WAE_encoder"))
    th.save(args.discriminator.state_dict(), args.model_saving_string("WAE_discriminator"))

    return


def calculate_train_losses(args):
    if args.separate_comp:
        with open(f"{args.data_folder}train_tensors_comp0_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
            train_tensors_comp0 = pkl.load(tensor_pkl)
        with open(f"{args.data_folder}train_tensors_comp1_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
            train_tensors_comp1 = pkl.load(tensor_pkl)
        train_tensors = [[train_tensors_comp0[i].to(args.device),
                          train_tensors_comp1[i].to(args.device)] for i in range(len(train_tensors_comp0))]

    else:
        with open(f"{args.data_folder}train_tensors_offline_{args.FEATS}.pkl", "rb") as tensor_pkl:
            train_tensors = pkl.load(tensor_pkl)
            train_tensors = [tensor.to(args.device) for i, tensor in enumerate(train_tensors) if i != 467 and i != 585]

    reconstruction_error, critic_scores = predict(args, train_tensors, "Calculating training error distribution")
    args.train_reconstruction_errors = reconstruction_error
    args.train_critic_scores = critic_scores
    return


def calculate_test_losses(args):
    all_test_tensors = []
    if args.separate_comp:
        for loop in range(args.END_LOOP + 1):
            t = []
            with open(f"{args.data_folder}test_tensors_comp0_{loop}_{args.FEATS}.pkl", "rb") as tensor_pkl:
                test_tensors = pkl.load(tensor_pkl)
                t.append(test_tensors)
            with open(f"{args.data_folder}test_tensors_comp1_{loop}_{args.FEATS}.pkl", "rb") as tensor_pkl:
                test_tensors = pkl.load(tensor_pkl)
                t.append(test_tensors)
            all_test_tensors.extend([[t[0][i].to(args.device), t[1][i].to(args.device)] for i in range(len(t[0]))])
    else:
        for loop in range(args.END_LOOP + 1):
            with open(f"{args.data_folder}test_tensors_{loop}_{args.FEATS}.pkl", "rb") as tensor_pkl:
                test_tensors = pkl.load(tensor_pkl)
                all_test_tensors.extend(test_tensors)

    reconstruction_errors, critic_scores = predict(args, all_test_tensors, "Testing on new data")

    results = {"test": {"reconstruction": reconstruction_errors,
                        "critic": critic_scores},
               "train": {"reconstruction": args.train_reconstruction_errors,
                         "critic": args.train_critic_scores}}

    with open(args.results_string("complete"), "wb") as loss_file:
        pkl.dump(results, loss_file)

    return


def load_parameters(arguments):
    FEATS_TO_NUMBER = {"analog_feats": 8, "digital_feats": 8, "all_feats": 16}

    arguments.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    arguments.FEATS = f"{arguments.FEATS}_feats"
    arguments.NUMBER_FEATURES = FEATS_TO_NUMBER[arguments.FEATS]

    arguments.results_folder = "results/"
    arguments.data_folder = "data/"

    arguments.model_string = lambda model: f"{model}_{arguments.MODEL_NAME}_{arguments.FEATS}_{arguments.EMBEDDING}_{arguments.LSTM_LAYERS}"

    print(f"Starting execution of model: {arguments.model_string('WAE')}")

    arguments.results_string = lambda loop_no: f"{arguments.results_folder}{loop_no}_losses_{arguments.model_string('WAE')}_{arguments.EPOCHS}_{arguments.LR}.pkl"
    arguments.model_saving_string = lambda model: f"{arguments.results_folder}offline_{arguments.model_string(model)}_{arguments.EPOCHS}_{arguments.LR}.pt"

    with open(f"{arguments.data_folder}online_train_val_test_inds.pkl", "rb") as indices_pkl:
        arguments.train_indices, arguments.val_indices, arguments.test_indices = pkl.load(indices_pkl)

    arguments.decoder = Decoder(arguments.EMBEDDING,
                                arguments.NUMBER_FEATURES,
                                arguments.DROPOUT,
                                arguments.LSTM_LAYERS).to(arguments.device)

    arguments.encoder = Encoder(arguments.NUMBER_FEATURES,
                                arguments.EMBEDDING,
                                arguments.DROPOUT,
                                arguments.LSTM_LAYERS).to(arguments.device)

    models = dict(SimpleDiscriminator=SimpleDiscriminator,
                  LSTMDiscriminator=LSTMDiscriminator,
                  ConvDiscriminator=ConvDiscriminator)

    arguments.discriminator = models[arguments.MODEL_NAME](arguments.EMBEDDING,
                                                           arguments.DROPOUT).to(arguments.device)

    return arguments


def main(arguments):
    if all([os.path.exists(arguments.model_saving_string(model)) for model in ["WAE_encoder",
                                                                               "WAE_decoder",
                                                                               "WAE_discriminator"]]) \
            and not arguments.force_training:

        arguments.decoder.load_state_dict(th.load(arguments.model_saving_string("WAE_decoder")))
        arguments.encoder.load_state_dict(th.load(arguments.model_saving_string("WAE_encoder")))
        arguments.discriminator.load_state_dict(th.load(arguments.model_saving_string("WAE_discriminator")))
    else:
        offline_train(arguments)

    calculate_train_losses(arguments)
    calculate_test_losses(arguments)


if __name__ == "__main__":
    argument_dict = parse_arguments()
    argument_dict = load_parameters(argument_dict)
    main(argument_dict)
