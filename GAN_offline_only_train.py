import os
import numpy as np
import torch as th
from torch import optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import pickle as pkl
import tqdm
from torch.autograd import Variable
from ArgumentParser import parse_arguments
from LSTM_GAN import Encoder, Decoder, CriticDecoder, CriticEncoder
import dtw


def train_critic_decoder(optimizer, train_tensor, random_latent_space, args):
    optimizer.zero_grad()
    decoded_real_x = args.critic_decoder(train_tensor)
    reconstructed_random_latent = args.decoder(random_latent_space)
    decoded_random_x = args.critic_decoder(reconstructed_random_latent)

    wasserstein_loss_real_x = th.mean(-th.ones(decoded_real_x.shape).to(args.device) * decoded_real_x)
    wasserstein_loss_random_x = th.mean(th.ones(decoded_random_x.shape).to(args.device) * decoded_random_x)

    random_weight_average_parameter = th.rand(train_tensor.shape).to(args.device)
    interpolates = Variable(random_weight_average_parameter * train_tensor +
                            (1 - random_weight_average_parameter) * reconstructed_random_latent)
    interpolates.requires_grad_(True)
    interpolates_critic = args.critic_decoder(interpolates)
    interpolates_critic.mean().backward()
    gradients = interpolates.grad
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.GP_hyperparam

    loss = gradient_penalty + wasserstein_loss_random_x + wasserstein_loss_real_x
    loss.backward()
    optimizer.step()

    return loss.item()


def train_critic_encoder(optimizer, train_tensor, random_latent_space, args):
    optimizer.zero_grad()
    latent_space = args.encoder(train_tensor)
    critic_real_latent = args.critic_encoder(latent_space)
    critic_random_latent = args.critic_encoder(random_latent_space)

    wasserstein_loss_real_x = th.mean(-th.ones(critic_real_latent.shape).to(args.device) * critic_real_latent)
    wasserstein_loss_random_x = th.mean(th.ones(critic_random_latent.shape).to(args.device) * critic_random_latent)

    # gradient penalty regularization

    random_weight_average_parameter = th.rand(latent_space.shape).to(args.device)
    interpolates = Variable(random_weight_average_parameter * latent_space +
                            (1 - random_weight_average_parameter) * random_latent_space)
    interpolates.requires_grad_(True)
    interpolates_critic = args.critic_encoder(interpolates)
    interpolates_critic.mean().backward()
    gradients = interpolates.grad
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.GP_hyperparam

    loss = gradient_penalty + wasserstein_loss_random_x + wasserstein_loss_real_x
    loss.backward()
    optimizer.step()

    return loss.item()


def train_model(train_tensors,
                epochs,
                lr,
                args):
    optimizer_critic_encoder = optim.Adam(args.critic_encoder.parameters(), lr=lr)
    optimizer_critic_decoder = optim.Adam(args.critic_decoder.parameters(), lr=lr)
    optimizer_encoder = optim.Adam(args.encoder.parameters(), lr=lr)
    optimizer_decoder = optim.Adam(args.decoder.parameters(), lr=lr)

    loss_over_time = {"critic_encoder": [], "critic_decoder": [],
                      "encoder": [], "decoder": []}

    multivariate_normal = MultivariateNormal(th.zeros(args.EMBEDDING), th.eye(args.EMBEDDING))

    for epoch in range(epochs):

        critic_tensors = [train_tensors[i] for i in np.random.randint(len(train_tensors), size=args.BATCH_SIZE)]
        for critic_iter in range(args.critic_iterations):
            critic_encoder_losses = []
            critic_decoder_losses = []
            with tqdm.tqdm(critic_tensors, unit="cycles") as tqdm_epoch:
                for critic_tensor in tqdm_epoch:
                    tqdm_epoch.set_description(f"Critics iteration {critic_iter + 1}")
                    random_latent_space = multivariate_normal.sample(critic_tensor.shape[:2]).to(args.device)
                    critic_encoder_losses.append(train_critic_encoder(optimizer_critic_encoder,
                                                                      critic_tensor,
                                                                      random_latent_space,
                                                                      args))

                    critic_decoder_losses.append(train_critic_decoder(optimizer_critic_decoder,
                                                                      critic_tensor,
                                                                      random_latent_space,
                                                                      args))

            loss_over_time["critic_encoder"].append(np.mean(critic_encoder_losses))
            loss_over_time["critic_decoder"].append(np.mean(critic_decoder_losses))

        print(f'Epoch {epoch + 1}: critic encoder loss {np.mean(critic_encoder_losses)} critic decoder loss {np.mean(critic_decoder_losses)}')

        encoder_losses = []
        decoder_losses = []
        with tqdm.tqdm(train_tensors, unit="cycles") as tqdm_epoch:
            for train_tensor in tqdm_epoch:
                tqdm_epoch.set_description(f"Reconstruction phase Epoch {epoch + 1}")
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()

                random_latent_space = multivariate_normal.sample(train_tensor.shape[:2]).to(args.device)
                latent_space = args.encoder(train_tensor)
                reconstructed_tensor = args.decoder(latent_space)

                critic_real_x = args.critic_decoder(train_tensor)
                reconstructed_random_latent = args.decoder(random_latent_space)
                critic_random_x = args.critic_decoder(reconstructed_random_latent)

                # Here we are minimizing the wasserstein loss (instead of maximizing as in the critic)
                # So we no longer swap the signs
                wl_real_critic_decoder = th.mean(th.ones(critic_real_x.shape).to(args.device) * critic_real_x)
                wl_random_critic_decoder = th.mean(-th.ones(critic_random_x.shape).to(args.device) * critic_random_x)

                critic_real_latent = args.critic_encoder(latent_space)
                critic_random_latent = args.critic_encoder(random_latent_space)

                wl_real_critic_encoder = th.mean(
                    th.ones(critic_real_latent.shape).to(args.device) * critic_real_latent)
                wl_random_critic_encoder = th.mean(
                    -th.ones(critic_random_latent.shape).to(args.device) * critic_random_latent)

                mse_loss = F.mse_loss(reconstructed_tensor, train_tensor)

                loss_encoder = wl_random_critic_encoder + wl_real_critic_encoder + mse_loss
                loss_decoder = wl_real_critic_decoder + wl_random_critic_decoder + mse_loss

                loss_decoder.backward(retain_graph=True)
                loss_encoder.backward()
                optimizer_decoder.step()
                optimizer_encoder.step()

                encoder_losses.append(loss_encoder.item())
                decoder_losses.append(loss_decoder.item())

        loss_over_time['encoder'].append(np.mean(encoder_losses))
        loss_over_time['decoder'].append(np.mean(decoder_losses))

        print(f'Epoch {epoch + 1}: encoder loss {np.mean(encoder_losses)} decoder loss {np.mean(decoder_losses)}')

    return loss_over_time


def calc_reconstruction_error(reconstructed_ts, original_ts, args):
    if args.reconstruction_error_metric == "mse":
        return F.mse_loss(reconstructed_ts, original_ts)

    return dtw.dtw(reconstructed_ts, original_ts, distance_only=True).normalizedDistance


def predict(args, test_tensors, tqdm_desc):
    reconstruction_errors = []
    critic_scores = []
    with th.no_grad():
        args.encoder.eval()
        args.decoder.eval()
        args.critic_decoder.eval()
        with tqdm.tqdm(test_tensors, unit="cycles") as tqdm_epoch:
            for test_tensor in tqdm_epoch:
                tqdm_epoch.set_description(tqdm_desc)
                if test_tensor.shape[1] > 3600:
                    reconstruction_errors.append(100 * max(reconstruction_errors))
                    critic_scores.append(0)
                test_tensor = test_tensor.to(args.device)
                reconstruction = args.decoder(args.encoder(test_tensor))
                reconstruction_errors.append(calc_reconstruction_error(reconstruction, test_tensor, args))
                try:
                    critic_score = args.critic_decoder(test_tensor)
                    critic_scores.append(critic_score)
                except th.cuda.OutOfMemoryError as e:
                    print(test_tensor.shape)
                    raise e
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

    with open(args.results_string("offline", args.reconstruction_error_metric), "wb") as loss_file:
        pkl.dump(loss_over_time, loss_file)

    th.save(args.decoder.state_dict(), args.model_saving_string("GAN_decoder"))
    th.save(args.encoder.state_dict(), args.model_saving_string("GAN_encoder"))
    th.save(args.critic_decoder.state_dict(), args.model_saving_string("GAN_critic_decoder"))
    th.save(args.critic_encoder.state_dict(), args.model_saving_string("GAN_critic_encoder"))

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

    with open(args.results_string("complete", args.reconstruction_error_metric), "wb") as loss_file:
        pkl.dump(results, loss_file)

    return


def load_parameters(arguments):
    FEATS_TO_NUMBER = {"analog_feats": 8, "digital_feats": 8, "all_feats": 16}

    arguments.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    arguments.FEATS = f"{arguments.FEATS}_feats"
    arguments.NUMBER_FEATURES = FEATS_TO_NUMBER[arguments.FEATS]

    arguments.results_folder = "results/"
    arguments.data_folder = "data/"

    arguments.model_string = lambda model: f"{model}_{arguments.FEATS}_{arguments.EMBEDDING}_{arguments.LSTM_LAYERS}"

    print(f"Starting execution of model: {arguments.model_string('GAN')}")

    arguments.results_string = lambda loop_no, reconstruction: f"{arguments.results_folder}{loop_no}_losses_{reconstruction}_{arguments.model_string('GAN')}_{arguments.EPOCHS}_{arguments.LR}.pkl"
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

    arguments.critic_encoder = CriticEncoder(arguments.EMBEDDING,
                                             arguments.LSTM_LAYERS,
                                             arguments.NHEADS,
                                             arguments.DROPOUT,
                                             arguments.device).to(arguments.device)

    arguments.critic_decoder = CriticDecoder(arguments.NUMBER_FEATURES,
                                             arguments.LSTM_LAYERS,
                                             arguments.NHEADS,
                                             arguments.DROPOUT).to(arguments.device)

    return arguments


def main(arguments):
    if all([os.path.exists(arguments.model_saving_string(model)) for model in ["GAN_encoder",
                                                                               "GAN_decoder",
                                                                               "GAN_critic_encoder",
                                                                               "GAN_critic_decoder"]]) \
            and not arguments.force_training:

        arguments.decoder.load_state_dict(th.load(arguments.model_saving_string("GAN_decoder")))
        arguments.encoder.load_state_dict(th.load(arguments.model_saving_string("GAN_encoder")))
        arguments.critic_encoder.load_state_dict(th.load(arguments.model_saving_string("GAN_critic_encoder")))
        arguments.critic_decoder.load_state_dict(th.load(arguments.model_saving_string("GAN_critic_decoder")))
    else:
        offline_train(arguments)

    calculate_train_losses(arguments)
    calculate_test_losses(arguments)


if __name__ == "__main__":
    argument_dict = parse_arguments()
    argument_dict = load_parameters(argument_dict)
    main(argument_dict)
