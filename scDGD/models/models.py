import torch
import torch.nn as nn
from scDGD.classes.output_distributions import NBLayer
import json



class DGD(nn.Module):
    def __init__(
        self,
        out,
        latent=20,
        hidden=[100, 100, 100],
        r_init=2,
        output_prediction_type="mean",
        output_activation="sigmoid"
    ):
        super(DGD, self).__init__()

        self.main = nn.ModuleList()

        if type(hidden) is not int:
            n_hidden = len(hidden)
            self.main.append(nn.Linear(latent, hidden[0]))
            self.main.append(nn.ReLU(True))
            for i in range(n_hidden - 1):
                self.main.append(nn.Linear(hidden[i], hidden[i + 1]))
                self.main.append(nn.ReLU(True))
            self.main.append(nn.Linear(hidden[-1], out))
        else:
            self.main.append(nn.Linear(latent, hidden))
            self.main.append(nn.ReLU(True))
            self.main.append(nn.Linear(hidden, out))

        self.nb = NBLayer(
            out,
            r_init=r_init,
            output=output_prediction_type,
            activation=output_activation
        )

    def forward(self, z):
        for i in range(len(self.main)):
            z = self.main[i](z)
        return self.nb(z)

    @classmethod
    def load(cls, save_dir="./"):
        # get saved hyper-parameters
        with open(save_dir + "dgd_hyperparameters.json", "r") as fp:
            param_dict = json.load(fp)

        model = cls(
            out=param_dict["n_genes"],
            latent=param_dict["latent"],
            hidden=param_dict["hidden"],
            r_init=param_dict["r_init"],
            scaling_type=param_dict["scaling_type"],
            output=param_dict["output"],
            activation=param_dict["activation"],
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(
            torch.load(
                save_dir + param_dict["name"] + "_decoder.pt", map_location=device
            )
        )
        return model
