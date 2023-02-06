import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from model import data2vec_base
from utils import update_momentum

'''
Dummy Dataset
'''
class D2VDataset(Dataset):
    def __init__(self,
                 dataset_path,
                 stage='train',
                 ):
        super().__init__()
        
        self.data = torch.randn(100, 3, 224, 224)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

'''
Placeholder for datamodule in pytorch lightning
'''
class D2VDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path,
                 batch_size=4,
                 num_workers=4,
                 pin_memory=True,
                 shuffle=True
                 ):
        super().__init__()
        
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        
    def setup(self, stage=None):
        self.train_dataset = D2VDataset(dataset_path=self.dataset_path, stage='train')
        self.val_dataset = D2VDataset(dataset_path=self.dataset_path, stage='val')
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

'''
pytorch lightning model
'''
class data2vec(pl.LightningModule):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3, 
            embed_dim=64,
            masking_ratio=0.5,
            heads=8,
            depth=8,
            decoder_depth=3,
            decoder_dim=64,
            decoder_kernel_size=3, 
            decoder_padding=1, 
            decoder_groups=1,
            post_emb_norm=True,
            dropout=0.,
            k=4, 
            lr=1e-3,
            M = 8, #number of different masked versions
            m=0.99, #momentum
    ):
        super().__init__()
        self.save_hyperparameters()
        
        #define models
        self.teacher = data2vec_base(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, masking_ratio=0., heads=heads, depth=depth, decoder_depth=decoder_depth, decoder_dim=decoder_dim, decoder_kernel_size=decoder_kernel_size, decoder_padding=decoder_padding, decoder_groups=decoder_groups, post_emb_norm=post_emb_norm, dropout=dropout, k=k, is_teacher=True)
        
        self.student = data2vec_base(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, masking_ratio=masking_ratio, heads=heads, depth=depth, decoder_depth=decoder_depth, decoder_dim=decoder_dim, decoder_kernel_size=decoder_kernel_size, decoder_padding=decoder_padding, decoder_groups=decoder_groups, post_emb_norm=post_emb_norm, dropout=dropout, k=k, is_teacher=False)
        
        #define hyperparameters
        self.m = m
        self.M = M
        self.lr = lr
        
        #define loss
        self.criterion = nn.MSELoss()
    
    def forward(self, x, encoder):
        return encoder(x)
    
    def training_step(self, batch, batch_idx):
        x = batch

        y_teacher = self(x, self.teacher) #get teacher embedding
        Y_student = torch.stack([self(x, self.student) for _ in range(self.M)], dim=0) #get student embeddings
    
        loss = self.criterion(Y_student, torch.stack([y_teacher for _ in range(self.M)], dim=0))
        self.log('train_loss', loss)
                    
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        y_teacher = self(x, self.teacher) #get teacher embedding
        Y_student = torch.stack([self(x, self.student) for _ in range(self.M)], dim=0) #get student embeddings
        
        loss = self.criterion(Y_student, torch.stack([y_teacher for _ in range(self.M)], dim=0))
        self.log('val_loss', loss)
        
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        return self(batch, self.teacher) #just get teacher embedding
    
    def on_after_backward(self):
        update_momentum(self.student, self.teacher, self.m)

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

if __name__ == '__main__':
    dataset = D2VDataModule(dataset_path='data')

    model = data2vec(img_size=224, patch_size=16, in_chans=3, embed_dim=8, masking_ratio=0.5, heads=4, depth=4, post_emb_norm=True, dropout=0., k=4, lr=1e-3, M = 2, m=0.99)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    trainer = pl.Trainer(
        accelerator='cpu',
        precision=16,
        max_epochs=10,
        callbacks=[lr_monitor, model_summary],
    )

    trainer.fit(model, dataset)
