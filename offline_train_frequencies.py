import os
import numpy as np
import torch as th
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import pickle as pkl
import tqdm
from ArgumentParser import parse_arguments
from LSTM_AAE import Encoder, Decoder, SimpleDiscriminator, LSTMDiscriminator, ConvDiscriminator
from torch.utils.data import Dataset, DataLoader


class FrequencyDataset(Dataset):
    def __init__(self, data_location):
        with open(data_location, "rb") as pklfile:
            self.data = pkl.load(pklfile)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ind):
        return self.data[ind, :, :]


####################
#
# Based on the implementation: https://github.com/schelotto/Wasserstein-AutoEncoders
#
####################

#th.autograd.set_detect_anomaly(True)


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def train_discriminator(optimizer_discriminator, multivariate_normal, epoch, args):

    frozen_params(args.encoder)
    frozen_params(args.decoder)
    free_params(args.discriminator)

    losses = []
    with tqdm.tqdm(args.train_dataloader, unit="batches") as tqdm_epoch:
        for train_batch in tqdm_epoch:
            tqdm_epoch.set_description(f"Discriminator Epoch {epoch + 1}")
            optimizer_discriminator.zero_grad()
            train_batch = train_batch.to(args.device)
            random_latent_space = multivariate_normal.sample((train_batch.shape[0], 1)).to(args.device)

            real_latent_space = args.encoder(train_batch).unsqueeze(1)

            discriminator_real = args.discriminator(real_latent_space)
            discriminator_random = args.discriminator(random_latent_space)

            loss_random_term = th.log(discriminator_random)
            loss_real_term = th.log(1-discriminator_real)

            loss = args.WAE_regularization_term * -th.mean(loss_real_term + loss_random_term)
            loss.backward()

            nn.utils.clip_grad_norm_(args.discriminator.parameters(), 1)
            optimizer_discriminator.step()
            losses.append(loss.item())

    return losses


def train_reconstruction(optimizer_encoder, optimizer_decoder, epoch, args):

    if args.use_discriminator:
        free_params(args.encoder)
        free_params(args.decoder)
        frozen_params(args.discriminator)

    losses = []
    with tqdm.tqdm(args.train_dataloader, unit="batches") as tqdm_epoch:
        for i, train_batch in enumerate(tqdm_epoch):
            tqdm_epoch.set_description(f"Encoder/Decoder Epoch {epoch + 1}")
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            train_batch = train_batch.to(args.device)
            real_latent_space = args.encoder(train_batch).unsqueeze(1)
            stacked_LV = real_latent_space.repeat(1, train_batch.shape[1], 1).to(args.device)

            reconstructed_input = args.decoder(stacked_LV)
            reconstruction_loss = F.mse_loss(reconstructed_input, train_batch)

            if args.use_discriminator:
                discriminator_real_latent = args.discriminator(real_latent_space)

                discriminator_loss = args.WAE_regularization_term * (th.log(discriminator_real_latent))

                loss = th.mean(reconstruction_loss - discriminator_loss)

            else:

                loss = reconstruction_loss

            loss.backward()

            nn.utils.clip_grad_norm_(args.encoder.parameters(), 1)
            nn.utils.clip_grad_norm_(args.decoder.parameters(), 1)

            optimizer_encoder.step()
            optimizer_decoder.step()
            losses.append(loss.item())

    return losses


def train_model(epochs,
                args):

    if args.use_discriminator:
        optimizer_discriminator = optim.Adam(args.discriminator.parameters(), lr=0.5 * args.LR)

    optimizer_encoder = optim.Adam(args.encoder.parameters(), lr=args.LR)
    optimizer_decoder = optim.Adam(args.decoder.parameters(), lr=args.LR)

    loss_over_time = {"discriminator": [], "encoder/decoder": []}

    multivariate_normal = MultivariateNormal(th.zeros(args.EMBEDDING), th.eye(args.EMBEDDING))

    for epoch in range(epochs):

        if args.use_discriminator:
            discriminator_losses = train_discriminator(optimizer_discriminator,
                                                       multivariate_normal, epoch, args)
            loss_over_time['discriminator'].append(np.mean(discriminator_losses))

        encoder_decoder_losses = train_reconstruction(optimizer_encoder, optimizer_decoder, epoch, args)
        loss_over_time['encoder/decoder'].append(np.mean(encoder_decoder_losses))

        if args.use_discriminator:
            print(f'Epoch {epoch + 1}: discriminator loss {np.mean(discriminator_losses)} encoder/decoder loss {np.mean(encoder_decoder_losses)}')
        else:
            print(f'Epoch {epoch + 1}: encoder/decoder loss {np.mean(encoder_decoder_losses)}')

    return loss_over_time


def predict(args, test_dataloader, tqdm_desc):
    reconstruction_errors = []
    critic_scores = []
    with th.no_grad():
        args.encoder.eval()
        args.decoder.eval()
        args.discriminator.eval()
        with tqdm.tqdm(test_dataloader, unit="cycles") as tqdm_epoch:
            for test_batch in tqdm_epoch:
                tqdm_epoch.set_description(tqdm_desc)
                test_batch = test_batch.to(args.device)
                latent_vector = args.encoder(test_batch).unsqueeze(1)

                stacked_LV = latent_vector.repeat(1, test_batch.shape[1], 1).to(args.device)

                reconstruction = args.decoder(stacked_LV)
                reconstruction_errors.append(F.mse_loss(reconstruction, test_batch).item())
                if args.use_discriminator:
                    critic_score = th.mean(args.discriminator(latent_vector))
                    critic_scores.append(critic_score.item())

    return reconstruction_errors, critic_scores


def offline_train(args):
    print(f"Starting offline training")

    loss_over_time = train_model(epochs=args.EPOCHS,
                                 args=args)

    if args.use_discriminator:
        results_string = args.results_string("offline", "WAE")
        model_string_decoder = args.model_saving_string("WAE_decoder")
        model_string_encoder = args.model_saving_string("WAE_encoder")
    else:
        results_string = args.results_string("offline", "AE")
        model_string_decoder = args.model_saving_string("AE_decoder")
        model_string_encoder = args.model_saving_string("AE_encoder")

    with open(results_string, "wb") as loss_file:
        pkl.dump(loss_over_time, loss_file)

    th.save(args.decoder.state_dict(), model_string_decoder)
    th.save(args.encoder.state_dict(), model_string_encoder)
    if args.use_discriminator:
        th.save(args.discriminator.state_dict(), args.model_saving_string("WAE_discriminator"))

    return


def calculate_train_losses(args):

    reconstruction_error, critic_scores = predict(args, args.train_scores, "Calculating training error distribution")
    args.train_reconstruction_errors = reconstruction_error
    args.train_critic_scores = critic_scores
    return


def calculate_test_losses(args):

    reconstruction_errors, critic_scores = predict(args, args.test_dataloader, "Testing on new data")

    results = {"test": {"reconstruction": reconstruction_errors,
                        "critic": critic_scores},
               "train": {"reconstruction": args.train_reconstruction_errors,
                         "critic": args.train_critic_scores}}

    if args.use_discriminator:
        results_string = args.results_string("complete", "WAE")
    else:
        results_string = args.results_string("complete", "AE")

    with open(results_string, "wb") as loss_file:
        pkl.dump(results, loss_file)

    return


def load_parameters(arguments):

    arguments.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    arguments.NUMBER_FEATURES = 100

    arguments.results_folder = "results/"
    arguments.data_folder = "data/"

    train_set = FrequencyDataset(arguments.train_tensor)
    test_set = FrequencyDataset(arguments.test_tensor)

    arguments.train_dataloader = DataLoader(train_set, batch_size=arguments.BATCH_SIZE, shuffle=True)
    arguments.train_scores = DataLoader(train_set, batch_size=1, shuffle=False)
    arguments.test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    arguments.model_string = lambda model: f"{model}_{arguments.MODEL_NAME}_frequencies_{arguments.sensor}_{arguments.EMBEDDING}_{arguments.LSTM_LAYERS}_{arguments.WAE_regularization_term}"

    if arguments.use_discriminator:
        print(f"Starting execution of model: {arguments.model_string('WAE')}")
    else:
        print(f"Starting execution of model: {arguments.model_string('AE')}")

    arguments.results_string = lambda loop_no, model_label: f"{arguments.results_folder}{loop_no}_losses_{arguments.model_string(model_label)}_{arguments.EPOCHS}_{arguments.LR}_{arguments.BATCH_SIZE}.pkl"
    arguments.model_saving_string = lambda model: f"{arguments.results_folder}offline_{arguments.model_string(model)}_{arguments.EPOCHS}_{arguments.LR}_{arguments.BATCH_SIZE}.pt"

    if arguments.use_discriminator:
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
                                                               arguments.DROPOUT,
                                                               n_layers=arguments.LSTM_LAYERS,
                                                               disc_hidden=arguments.disc_hidden,
                                                               kernel_size=arguments.tcn_kernel).to(arguments.device)

    else:
        arguments.encoder = Encoder(arguments.NUMBER_FEATURES,
                                    arguments.EMBEDDING,
                                    arguments.DROPOUT,
                                    arguments.LSTM_LAYERS).to(arguments.device)

        arguments.decoder = Decoder(arguments.EMBEDDING,
                                    arguments.NUMBER_FEATURES,
                                    arguments.DROPOUT,
                                    arguments.LSTM_LAYERS).to(arguments.device)

        arguments.discriminator = None

    return arguments


def main(arguments):

    if arguments.use_discriminator:
        models_exist = all([os.path.exists(arguments.model_saving_string(model)) for model in ["WAE_encoder",
                                                                                               "WAE_decoder",
                                                                                               "WAE_discriminator"]])
    else:
        models_exist = all([os.path.exists(arguments.model_saving_string(model)) for model in ["AE_encoder",
                                                                                               "AE_decoder"]])

    if models_exist and not arguments.force_training:
        if arguments.use_discriminator:
            arguments.decoder.load_state_dict(th.load(arguments.model_saving_string("WAE_decoder")))
            arguments.encoder.load_state_dict(th.load(arguments.model_saving_string("WAE_encoder")))
            arguments.discriminator.load_state_dict(th.load(arguments.model_saving_string("WAE_discriminator")))
        else:
            arguments.decoder.load_state_dict(th.load(arguments.model_saving_string("AE_decoder")))
            arguments.encoder.load_state_dict(th.load(arguments.model_saving_string("AE_encoder")))

    else:
        offline_train(arguments)

    calculate_train_losses(arguments)
    calculate_test_losses(arguments)


if __name__ == "__main__":
    argument_dict = parse_arguments()
    argument_dict = load_parameters(argument_dict)
    main(argument_dict)
