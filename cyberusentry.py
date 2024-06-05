import torch.nn as nn
import torch
import pickle

from uccs import gallery_similarity


class CyberuSentry(nn.Module):
    """Class CyberuSentry implementing the cherry on top
    of this whole repository. CyberuSentry model is composed
    of three individual heads, "cer", "ber", and "os".

    For the purposes of UCCS Challenge, each individual head
    needs to come with its own gallery corresponding to UCCS
    identities.

    In the future, gallery-less functionality should also be
    possible.
    """

    def __init__(
        self,
        h1_path: str,
        h1_class_instance: nn.Module,
        h1_gallery: str,
        h2_path: str,
        h2_class_instance: nn.Module,
        h2_gallery: str,
        h3_path: str,
        h3_class_instance: nn.Module,
        h3_gallery: str,
    ) -> None:
        super().__init__()
        self.device = "cpu"

        # CER
        self.cer_path = h1_path
        self.cer = h1_class_instance
        self.cer.load_state_dict(torch.load(self.cer_path, map_location=self.device))
        self.cer.eval()
        with open(h1_gallery, "rb") as cer_gal:
            self.cer_gallery = pickle.load(cer_gal)

        # BER
        self.ber_path = h2_path
        self.ber = h2_class_instance
        self.ber.load_state_dict(torch.load(self.ber_path, map_location=self.device))
        self.ber.eval()
        with open(h2_gallery, "rb") as ber_gal:
            self.ber_gallery = pickle.load(ber_gal)

        # OS
        self.os_path = h3_path
        self.os = h3_class_instance
        self.os.load_state_dict(torch.load(self.os_path, map_location=self.device))
        self.os.eval()
        with open(h3_gallery, "rb") as os_gal:
            self.os_gallery = pickle.load(os_gal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on the provided input.
        Input is expected to be a cropped face converted
        to a model format. Face is run through each head,
        similarity scores against the UCCS gallery are
        computed and the average of them is returned back.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, expected to be a cropped face

        Returns
        -------
        torch.Tensor
            Tensor of shape [1, num_identities] where
            'num_identities' is the number of indentities
            in the gallery. Unlike standard Cosine Similarity,
            The values are in [0, 1] range.
        """
        with torch.no_grad():
            cer_out = self.cer(x)
            # Head1 EvaTinyEuclid =     ..., 1.75, 0.4
            # Head5 MobileNetV2Euclid = ..., 8.5,  0.6
            cer_out = gallery_similarity(
                self.cer_gallery, cer_out, "euclidean", 1.75, 0.4
            )

            ber_out = self.ber(x)
            ber_out = gallery_similarity(self.ber_gallery, ber_out, "cosine")

            os_out = self.os(x)
            os_out = gallery_similarity(self.os_gallery, os_out, "cosine")

            stacked_preds = torch.stack([cer_out, ber_out, os_out])
            final_preds = torch.mean(stacked_preds, dim=0)

        return final_preds
