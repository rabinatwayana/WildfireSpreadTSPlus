from typing import Any
import torch
import os
from .BaseModel import BaseModel
from .utae_paps_models.utae import UTAE

"""
RT: Major changes, common UTAE instead if UTAE and UTAEDumb
"""


class UTAELightningRabina(BaseModel):
    """_summary_ U-Net architecture with temporal attention in the bottleneck and skip connections.
    RT: flatten_temporal_dimension is not applicable in UTAE
    """
    def __init__(
        self,
        n_channels: int,
        # flatten_temporal_dimension: bool,
        pos_class_weight: float,
        encoder_weights=None,
        temporal_position_mode: str | None = None,  # RT:None, doy, relative, None is for flatten_temporal_dimension, but it can not be None in UTAE
        pretrained_checkpoint_path: str | None = None,
        *args: Any,
        **kwargs: Any
    ):
        # use_doy = temporal_position_mode == "doy"
        super().__init__(
            n_channels=n_channels,
            # flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            temporal_position_mode=temporal_position_mode,
            # use_doy=use_doy, temporal_position_mode instead of doy
            *args,
            **kwargs
        )
        self.temporal_position_mode = temporal_position_mode

        print(f"\n Using temporal_position_mode {self.temporal_position_mode }")

        self.model = UTAE(
            input_dim=n_channels,
            encoder_widths=[64, 64, 64, 128],
            decoder_widths=[32, 32, 64, 128],
            out_conv=[32, 1],
            str_conv_k=4,
            str_conv_s=2,
            str_conv_p=1,
            agg_mode="att_group",
            encoder_norm="group",
            n_head=16,
            d_model=256,
            d_k=4,
            encoder=False,
            return_maps=False,
            pad_value=0,
            padding_mode="reflect",
        )
        encoder_weights = encoder_weights if encoder_weights != "none" else None
        print(f"\n Using encoder_weights: {encoder_weights}")
        if encoder_weights == "pastis":
            print(f"\n Loading PASTIS checkpoint from {pretrained_checkpoint_path}")

            # repo_default = os.path.join(
            #     os.path.dirname(__file__), "utae_paps_models", "model.pth.tar"
            # )
            # env_ckpt = os.environ.get("UTAE_PASTIS_CKPT")
            # candidate_paths = [
            #     pretrained_checkpoint_path,
            #     env_ckpt,
            #     '/develop/data/utae_pre/model.pth.tar',
            #     repo_default,
            #     '/home/sl221120/WildfireSpreadTS/src/models/utae_paps_models/model.pth.tar',
            # ]
            # pretrained_checkpoint = next(
            #     (path for path in candidate_paths if path and os.path.exists(path)),
            #     None,
            # )
            # if pretrained_checkpoint is None:
            #     raise FileNotFoundError(
            #         "PASTIS checkpoint not found. Set model.init_args.pretrained_checkpoint_path "
            #         "or export UTAE_PASTIS_CKPT."
            #     )
            # self.load_checkpoint(pretrained_checkpoint)
            if pretrained_checkpoint_path is None:
                raise FileNotFoundError(
                    "PASTIS checkpoint not found. Set model.init_args.pretrained_checkpoint_path "
                    "or export UTAE_PASTIS_CKPT."
                )
            self.load_checkpoint(pretrained_checkpoint_path)


    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a pretrained checkpoint for the model.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["state_dict"]
        prefix = "encoder."
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_state_dict[key[len(prefix):]] = value
            else:
                new_state_dict[key] = value
        model_state_dict = self.model.state_dict()
        filtered_state_dict = {
            k: v
            for k, v in new_state_dict.items()
            if k in model_state_dict and model_state_dict[k].size() == v.size()
        }
        # Load the weights into the model
        self.model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Checkpoint loaded successfully from '{checkpoint_path}'")

    def _get_temporal_positions(self, x: torch.Tensor, doys: torch.Tensor | None) -> torch.Tensor:
        # if not self.flatten_temporal_dimension:
            if self.temporal_position_mode == "relative":
                bsz, steps, _, _, _ = x.shape
                return torch.arange(steps, device=x.device).unsqueeze(0).repeat(bsz, 1)
            elif self.temporal_position_mode == "doy":
                if doys is None:
                    raise ValueError(
                        "temporal_position_mode='doy' requires DOY inputs from the dataloader."
                    )
                return doys
            else:
                raise ValueError(
                    f"Unknown temporal_position_mode='{self.temporal_position_mode}' for UTAE Model"
                )
        # else:
        #     raise ValueError(
        #         "Temporal positions cannot be computed because flatten_temporal_dimension is True"
        #     )

    # override forward function of BaseModel
    def forward(self, x: torch.Tensor, doys: torch.Tensor | None = None) -> torch.Tensor:
        # print(f"\n Using overidden forward pass of UTAE")
        batch_positions = self._get_temporal_positions(x, doys)
        return self.model(x, batch_positions=batch_positions, return_att=False)
