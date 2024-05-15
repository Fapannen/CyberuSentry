import torch.nn as nn
import torch
import pickle

from uccs import gallery_similarity


class CyberuSentry(nn.Module):
    def __init__(
        self,
        h1_path,
        h1_class_instance,
        h1_gallery,
        h2_path,
        h2_class_instance,
        h2_gallery,
        h3_path,
        h3_class_instance,
        h3_gallery,
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
        self.os.eval()
        self.os.load_state_dict(torch.load(self.os_path, map_location=self.device))
        with open(h3_gallery, "rb") as os_gal:
            self.os_gallery = pickle.load(os_gal)

    def forward(self, x):
        with torch.no_grad():
            cer_out = self.cer(x)
            cer_out = gallery_similarity(
                self.cer_gallery, cer_out, "euclidean", 1.75, 0.4, 4
            )

            ber_out = self.ber(x)
            ber_out = gallery_similarity(self.ber_gallery, ber_out, "cosine")

            os_out = self.os(x)
            os_out = gallery_similarity(self.os_gallery, os_out, "cosine")

            stacked_preds = torch.stack([cer_out, ber_out, os_out])
            final_preds = torch.mean(stacked_preds, dim=0)

        return final_preds
