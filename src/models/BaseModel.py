"""
RT Improvements:
Added sync_dist=True in all metrics and loss
added train_avg_precision
loss, instead of loss.item(), as loss.item() turns the tensor into a plain Python number, so Lightning can no longer treat it like a tensor for distributed reduction the same way.
added f1_threshild for test metrics

get_pred_and_gt() now accepts both (x, y) and (x, y, doys) instead of wrongly forcing all non-flattened models to use doys.
repeated-crop inference now supports both 4D and 5D inputs.
predict_step() now handles doys correctly and extracts the active-fire channel safely for both 4D and 5D inputs.
duplicate test_loss logging was removed.
test metrics are now logged once, explicitly, at epoch level.
focal loss no longer uses the broken 1 - pos_class_weight alpha; it now uses alpha_focal if provided, otherwise 0.25.
the duplicate/unused PR-curve figure creation was cleaned up.

"""

import math
from abc import ABC
from typing import Any, Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import wandb
import numpy as np
import torchvision.transforms.functional as TF
from segmentation_models_pytorch.losses import (DiceLoss, JaccardLoss,
                                                LovaszLoss)
from torchvision.ops import sigmoid_focal_loss
import matplotlib.pyplot as plt
import os

class BaseModel(pl.LightningModule, ABC):
    """_summary_ Base model class for all models in this project. Implements the training, validation and test steps, 
    as well as the loss function. 

    """
    def __init__(
        self,
        n_channels: int,
        pos_class_weight: float,
        loss_function: Literal["BCE", "Focal", "Lovasz", "Jaccard", "Dice"],
        flatten_temporal_dimension: bool = False,
        temporal_position_mode: str | None = None,
        # use_doy: bool = False, #RT
        crop_before_eval: bool = False,
        required_img_size: Optional[Tuple[int, int]] = None,
        alpha_focal: Optional[float] = None, 
        f1_threshold: Optional[float] = None,
        *args: Any,
        **kwargs: Any
    ):
        """_summary_ 

        Args:
            n_channels (int): _description_ Number of feature channels in the input data. Usually means number of features per time step, 
            except for U-Net which flattens the temporal dimension and uses this parameter as the total number of features. 
            flatten_temporal_dimension (bool): _description_ Whether to flatten the temporal dimension of the input data.
            pos_class_weight (float): _description_ Weight of the positive class in the loss function (only used for BCE and Focal loss).
            loss_function (Literal[&#39;BCE&#39;, &#39;Focal&#39;, &#39;Lovasz&#39;, &#39;Jaccard&#39;, &#39;Dice&#39;]): _description_ Which loss function to use. 
            # RT:REMOVE use_doy (bool, optional): _description_. Whether to use the doy of year (doy) as an additional input feature. Defaults to False.
            required_img_size (Optional[Tuple[int,int]], optional): _description_. Defaults to None. 
            When using a model that requires a specific image size, this parameter can be used to indicate it. We assume models require square images, 
            so this parameter indicates the side length. If set, the forward method will perform repeated inference on crops of the 
            image, and aggregate the results. This also works for non-square images. 
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        print("Using temporal_position_mode = ",self.hparams.temporal_position_mode)
        print("Using loss function = ",self.hparams.loss_function)
        print("Using positive class weight = ",self.hparams.pos_class_weight)


        # self.hparams.use_doy = use_doy #RT: As use_doy was not retained.
        # self.hparams.temporal_position_mode=temporal_position_mode

        if required_img_size is not None:
            self.hparams.required_img_size = torch.Size(
                required_img_size, device=self.device
            )

        # Normalize class weights by assuming that the negative class has weight 1
        if self.hparams.loss_function == "Focal":
            if self.hparams.alpha_focal is None:
                w = self.hparams.pos_class_weight
                alpha = w / (1 + w) if w is not None else 0.75
                self.hparams.alpha_focal = min(alpha, 0.95)
        #     self.hparams.pos_class_weight /= 1 + self.hparams.pos_class_weight  # RT: Calculating alpha for focal loss, alpha should be between 0-1

        self.loss = self.get_loss()
        threshold = self.hparams.f1_threshold if self.hparams.f1_threshold is not None else 0.5
        print(f"\n Using Threshold in metric calculation: {threshold}" )
        self.train_f1 = torchmetrics.F1Score("binary", threshold=threshold)

        # if self.hparams.f1_threshold:
            # self.train_f1 = torchmetrics.F1Score("binary", threshold=self.hparams.f1_threshold)
        # else:
        #     self.train_f1 = torchmetrics.F1Score("binary")
        self.val_f1 = self.train_f1.clone()
        self.test_f1 = self.train_f1.clone()

        self.train_avg_precision = torchmetrics.AveragePrecision("binary")
        self.val_avg_precision = self.train_avg_precision.clone()
        self.test_avg_precision = self.train_avg_precision.clone()

        # self.test_avg_precision = torchmetrics.AveragePrecision("binary")
        # self.val_avg_precision = self.test_avg_precision.clone()
        self.test_precision = torchmetrics.Precision("binary", threshold=threshold)
        self.test_recall = torchmetrics.Recall("binary", threshold=threshold)
        self.test_iou = torchmetrics.JaccardIndex("binary",threshold=threshold)
        self.conf_mat = torchmetrics.ConfusionMatrix("binary",threshold=threshold)

        # Plot PR curve at the end of training. Use fixed number of threshold to avoid the plot becoming 800MB+. 
        self.test_pr_curve = torchmetrics.PrecisionRecallCurve("binary", thresholds=100)

    def forward(self, x, doys=None):
    # def forward(self, x):
        # If doys are used, the model needs to re-implement the forward method
        if self.hparams.flatten_temporal_dimension and len(x.shape) == 5:
            x = x.flatten(start_dim=1, end_dim=2)
        return self.model(x)

    def get_pred_and_gt(self, batch):
        """_summary_ Unbatch the data and perform inference on each sample.

        Args:
            batch (_type_): _description_ Either a tuple of (x, y) or (x, y, doys).

        Raises:
            ValueError: _description_ If the batch size is not 1 and the model requires repeated inference on crops of the image. 
            This is the case for ConvLSTM, when predicting on the test set. During training, it uses random crops of the required size,
            so larger batch sizes can be used. 

        Returns:
            _type_: _description_ Prediction and ground truth for each sample in the batch.
        """

        # UTAE and TSViT use an additional doy feature as input. 
        # if self.hparams.use_doy:
        # if getattr(self.hparams, "use_doy", False): #RT
        #     x, y, doys = batch
        # else:
        #     x, y = batch
        #     doys = None
        
        if len(batch) == 3:
            x, y, doys = batch
        elif len(batch) == 2:
            x, y = batch
            doys = None
        else:
            raise ValueError(
                f"Unexpected batch structure with {len(batch)} elements. "
                "Expected either (x, y) or (x, y, doys)."
            )


        # If the model requires a certain fixed size, perform repeated inference on crops of the image,
        # and aggregate the results. When we reach the last row or column, which might not be divisible by
        # the required size, we align the crop window with the right/bottom edge of the image. This means 
        # that there is some amount of overlap between the last two crops in each row/column. We handle this
        # by simply overwriting the existing predictions with the new ones. 

        if self.hparams.required_img_size is not None:
            if x.ndim == 5:
                B, _, _, H, W = x.shape
            elif x.ndim == 4:
                B, _, H, W = x.shape
            else:
                raise ValueError(
                    f"Expected 4D or 5D input for repeated cropping, got shape {tuple(x.shape)}."
                )

            if x.shape[-2:] != self.hparams.required_img_size:
                if B != 1:
                    raise ValueError(
                        "Not implemented: repeated cropping for batch size > 1."
                    )
                H_req, W_req = self.hparams.required_img_size

                n_H = math.ceil(H / H_req)
                n_W = math.ceil(W / W_req)

                agg_output = torch.zeros(B, H, W, device=self.device)

                for i in range(n_H):
                    for j in range(n_W):
                        if i == n_H - 1:
                            H1 = H - H_req
                            H2 = H
                        else:
                            H1 = i * H_req
                            H2 = (i + 1) * H_req
                        if j == n_W - 1:
                            W1 = W - W_req
                            W2 = W
                        else:
                            W1 = j * W_req
                            W2 = (j + 1) * W_req

                        if x.ndim == 5:
                            x_crop = x[:, :, :, H1:H2, W1:W2]
                        else:
                            x_crop = x[:, :, H1:H2, W1:W2]

                        agg_output[:, H1:H2, W1:W2] = self(x_crop, doys).squeeze(1)

                y_hat = agg_output
                return y_hat, y

        y_hat = self(x, doys).squeeze(1)

        return y_hat, y

    def center_crop(self, x, y, crop_size=128):
        """_summary_ Crops the center of the image to 128x128, 
        Only used for computing the test performance.

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        H_new, W_new = crop_size, crop_size
        x = TF.center_crop(x, (H_new, W_new))
        y = TF.center_crop(y, (H_new, W_new))
        
        return x, y


    def training_step(self, batch, batch_idx):
        """_summary_ Compute predictions and loss for the given batch. Log training loss and F1 score.

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_hat, y = self.get_pred_and_gt(batch)
        # print("y_hat statistics")
        # print(y_hat.min().item(), y_hat.max().item(), y.float().mean().item())


        if self.hparams.crop_before_eval:
            #print("Center cropping before eval...")
            y_hat, y = self.center_crop(y_hat, y)

        loss = self.compute_loss(y_hat, y)
        if not torch.isfinite(loss):
            print("Non-finite loss", loss)

        self.train_f1(y_hat, y)
        self.train_avg_precision(y_hat, y)
        self.log(
            "train_loss",
            # loss.item(),
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_avg_precision",
            self.train_avg_precision,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_f1",
            self.train_f1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """_summary_ Compute predictions and loss for the given batch. Log validation loss, AP, and F1 score.

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_hat, y = self.get_pred_and_gt(batch)

        if self.hparams.crop_before_eval:
            y_hat, y = self.center_crop(y_hat, y)

        loss = self.compute_loss(y_hat, y)
        # changing to val ap to match test metric
        self.val_avg_precision(y_hat, y)
        self.val_f1(y_hat, y)
        self.log(
            "val_loss",
            loss,
            # loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_avg_precision",
            self.val_avg_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_f1",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )  
        return loss

    def test_step(self, batch, batch_idx):
        """_summary_ Compute predictions and loss for the given batch. Log test loss, F1, AP, precision, recall, IoU and confusion matrix.

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_hat, y = self.get_pred_and_gt(batch)

        if self.hparams.crop_before_eval:
            y_hat, y = self.center_crop(y_hat, y)

        loss = self.compute_loss(y_hat, y)
        self.test_f1(y_hat, y)
        self.test_avg_precision(y_hat, y)
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_iou(y_hat, y)
        self.test_pr_curve.update(y_hat, y)
        self.conf_mat.update(y_hat, y)

        self.log(
            "test_loss",
            # loss.item(),
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log_dict(
            {
                "test_f1": self.test_f1,
                "test_AP": self.test_avg_precision,
                "test_precision": self.test_precision,
                "test_recall": self.test_recall,
                "test_iou": self.test_iou,

            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return loss

    def on_test_epoch_end(self) -> None:
        """_summary_ Log the test PR curve and confusion matrix after predicting all test samples.
        """
        #conf_mat = self.conf_mat.compute().cpu().numpy()
        #wandb_table = wandb.Table(
        #    data=conf_mat, columns=["PredictedBackground", "PredictedFire"]
        #)
        #wandb.log({"Test confusion matrix": wandb_table})

        precision, recall, thresholds = self.test_pr_curve.compute()

        # # Move tensors to CPU
        precision = precision.cpu().numpy()
        recall = recall.cpu().numpy()
        thresholds = thresholds.cpu().numpy()
        output_npz_path = os.path.join(self.trainer.default_root_dir, f"test_pr_curve.npz")
        # np.savez("test_pr_curve_data.npz", precision=precision, recall=recall, thresholds=thresholds)
        np.savez(output_npz_path, precision=precision, recall=recall, thresholds=thresholds)


        fig, ax = plt.subplots()
        ax.plot(recall, precision, marker='.')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        output_png_path = os.path.join(self.trainer.default_root_dir, f"test_pr_curve.png")
        fig.savefig(output_png_path, dpi=300, bbox_inches="tight")
        
        if wandb.run is not None:
            wandb.log({"Test PR Curve": wandb.Image(fig)})
        plt.close(fig)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 3:
            x, y, doys = batch
        elif len(batch) == 2:
            x, y = batch
            doys = None
        else:
            raise ValueError(
                f"Unexpected batch structure with {len(batch)} elements. "
                "Expected either (x, y) or (x, y, doys)."
            )

        if x.ndim == 5:
            x_af = x[:, -1, -1, :, :]
        elif x.ndim == 4:
            x_af = x[:, -1, :, :]
        else:
            raise ValueError(f"Unexpected input shape in predict_step: {tuple(x.shape)}")

        y_hat = self(x, doys).squeeze(1)
        return x_af, y, y_hat

    def get_loss(self):
        if self.hparams.loss_function == "BCE":
            return nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([self.hparams.pos_class_weight])
                # pos_weight=torch.Tensor(
                #     [self.hparams.pos_class_weight], device=self.device
                # )
            )
        elif self.hparams.loss_function == "Focal":
            return sigmoid_focal_loss
        elif self.hparams.loss_function == "Lovasz":
            return LovaszLoss(mode="binary")
        elif self.hparams.loss_function == "Jaccard":
            return JaccardLoss(mode="binary")
        elif self.hparams.loss_function == "Dice":
            return DiceLoss(mode="binary")

    # def compute_loss(self, y_hat, y):
    #     if self.hparams.loss_function == "Focal":
    #         alpha = self.hparams.alpha_focal if self.hparams.alpha_focal is not None else 0.25
    #         return self.loss(
    #             y_hat,
    #             y.float(),
    #             alpha=alpha,
    #             gamma=2,
    #             reduction="mean",
    #         )
    #     else:
    #         return self.loss(y_hat, y.float())
        

    def compute_loss(self, y_hat, y):
        if self.hparams.loss_function == "Focal":
            return self.loss(
                y_hat,
                y.float(),
                alpha=self.hparams.alpha_focal, #RT
                # alpha=1 - self.hparams.pos_class_weight, 
                gamma=2,
                reduction="mean",
            )
        else:
            return self.loss(y_hat, y.float())
