import torch
import torch.nn as nn
import torch.nn.functional as F

from pvae.models_utils import _weight_init

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VariationalEncoder(nn.Module):
    """
    Taken and adapted from https://avandekleut.github.io/vae/
    """

    def __init__(self, k, n_features):
        super().__init__()
        self.linear1 = nn.Linear(n_features, 512)
        _weight_init(self.linear1.weight)
        self.batch_norm1 = nn.BatchNorm1d(self.linear1.out_features)
        self.dropout1 = nn.Dropout(0.2)

        self.mu_layer = nn.Linear(512, k)
        _weight_init(self.mu_layer.weight)

        self.sigma_layer = nn.Linear(512, k)
        _weight_init(self.sigma_layer.weight)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.linear1(x)))
        x = self.dropout1(x)
        mu = self.mu_layer(x)
        sigma = torch.exp(self.sigma_layer(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
        return z

    def _apply(self, fn):
        super()._apply(fn)
        self.N.loc = fn(self.N.loc)
        self.N.scale = fn(self.N.scale)
        return self


class Decoder(nn.Module):
    def __init__(self, k, n_features):
        super().__init__()

        self.linear1 = nn.Linear(k, 512)
        _weight_init(self.linear1.weight)
        self.batch_norm1 = nn.BatchNorm1d(self.linear1.out_features)

        self.linear_recon = nn.Linear(512, n_features)
        _weight_init(self.linear_recon.weight)

    def forward(self, z):
        # predict input data from latent space
        x = F.relu(self.batch_norm1(self.linear1(z)))
        return self.linear_recon(x)

# Created a linear Decoder 
class LinearDecoder(nn.Module):
    def __init__(self, k, n_features):
        super().__init__()
        self.linear1 = nn.Linear(k, 512)
        _weight_init(self.linear1.weight)
        # self.batch_norm1 = nn.BatchNorm1d(self.linear1.out_features)

        self.linear_recon = nn.Linear(512, n_features)
        _weight_init(self.linear_recon.weight)
        
        # self.linear_recon = nn.Linear(k, n_features)
        # _weight_init(self.linear_recon.weight)

    def forward(self, z):
        # predict input data from latent space
        x = F.relu(self.linear1(z))
        return self.linear_recon(x)

# Variational Autoencoder since just setting the weight to pathway loss is not enough it can still effect the weights of the edges in the network
class VariationalAutoencoder(nn.Module):
    """
    Args:
        k: latent space dimension
        n_features: number of features in the input data (genes)
        n_samples: number of samples in the input data
        n_pathways: number of pathways
        use_logistic_regression: if True, use logistic regression for the pathway prediction
        kl_l: weight for the KL loss
        pred_l: weight for the pathway prediction loss
    """

    def __init__(
        self,
        k,
        n_features,
        n_samples,
        n_pathways,
        use_logistic_regression=False,
        kl_l: float = None,
        pred_l: float = None,
    ):
        super().__init__()
        self.encoder = VariationalEncoder(k, n_features)
        self.decoder = Decoder(k, n_features)
        # FIXME: remove use_logistic_regression here, and instead have anotehr class implementing
        #  the logistic regression predictor
        self.kl_l = kl_l
        self.pred_l = pred_l

    def forward(self, x, sample_indices=None):
        z = self.encoder(x)
        x_prime = self.decoder(z)
        pathways_pred = None
        return x_prime

    def training_step(
        self,
        batch,
        batch_idx,
        batch_data_idxs,
        pathways_train,
        pathways_val,
    ):
        """
        Training step for the model.

        Args:
            batch: training data
            batch_idx: global index of the batch
            batch_data_idxs: indexes of the samples in the batch; this is used to build the data matrix to train the
                pathway predictor
            pathways_train: pathways to predict
            pathways_val: pathways to use for validation

        Returns:
            A tuple with two elements: the total loss (Tensor), and a dictionary with the individual losses
            values (each as a Tensor).
        """
        x, sample_indices = batch, batch_data_idxs
        x_prime = self.forward(x, sample_indices)
        mse_loss = F.mse_loss(x_prime, x, reduction="sum")
        unweighted_loss = mse_loss + self.encoder.kl #+ pathways_loss

        train_losses = {
            "full": unweighted_loss.detach().cpu(),
            "mse": mse_loss.detach().cpu(),
            "kl": self.encoder.kl.detach().cpu(),
        }

        return (
            (mse_loss + self.encoder.kl * self.kl_l ),
            train_losses,
        )

    def validation_step(self, batch, global_batch_idx):
        """
        Validation step for the model.

        Args:
            batch: validation data
            global_batch_idx: global index of the batch
            debug: if True, print debug info

        Returns:
            A tuple with the output of the model and the loss.
        """
        x = batch
        x_prime = self.forward(x)

        mse_loss = F.mse_loss(x_prime, x, reduction="sum")

        unweighted_loss = mse_loss + self.encoder.kl

        val_losses = {
            "full": unweighted_loss.detach().cpu(),
            "mse": mse_loss.detach().cpu(),
            "kl": self.encoder.kl.detach().cpu(),
        }

        return x_prime, val_losses
