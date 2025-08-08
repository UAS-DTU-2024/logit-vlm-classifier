import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import h5py
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.HookedLVLM import HookedLVLM
import random
from tqdm import tqdm
import math
import ast

model = HookedLVLM(device='cpu',quantize=False) 

class MLLMDataset(Dataset):
    def __init__(self, image_dir, csv_path, embeddings_file=None, layer_range=(20, 33), transform=None):
        self.image_dir = image_dir
        self.embeddings_file = embeddings_file
        self.layer_range = layer_range
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()

        ])

        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.class_1_df = self.df[self.df['trauma'] == 1].copy()
        self.class_0_df = self.df[self.df['trauma'] == 0].copy()
        self.num_class_1 = len(self.class_1_df)
        self.current_df = None 
        self.resample()
        self.df = self.filter_existing_images()
        self.embeddings_h5 = None
        self.available_embeddings = set()  
        if embeddings_file and os.path.exists(embeddings_file):
            self.load_embeddings_file(embeddings_file)
            print(f"Loaded embeddings from: {embeddings_file}")
        else:
            print("No embeddings file provided - will return None for hidden states")
    
    def resample(self):
        sampled_class_0 = self.class_0_df.sample(n=self.num_class_1, replace=False)
        combined = pd.concat([self.class_1_df, sampled_class_0], axis=0)
        self.current_df = combined.sample(frac=1).reset_index(drop=True) 

    def load_embeddings_file(self, embeddings_file):
        
        try:
            self.embeddings_h5 = h5py.File(embeddings_file, 'r')

            if 'embeddings' in self.embeddings_h5:
                # Get all available embedding keys
                self.available_embeddings = set(self.embeddings_h5['embeddings'].keys())
                print(f"Found embeddings for {len(self.available_embeddings)} images")
                # print(f"Sample embedding keys: {list(self.available_embeddings)[:5]}")
            else:
                print("Warning: No 'embeddings' group found in HDF5 file")

        except Exception as e:
            print(f"Error loading embeddings file: {e}")
            self.embeddings_h5 = None


    def filter_existing_images(self):

        if not os.path.exists(self.image_dir):
            print(f"ERROR: Image directory does not exist: {self.image_dir}")
            return pd.DataFrame()

        existing_rows = []
        for idx, row in self.df.iterrows():
            image_name = str(row['image_name']).strip()
            image_path = os.path.join(self.image_dir, image_name)

            if os.path.exists(image_path):
                existing_rows.append(row)

        filtered_df = pd.DataFrame(existing_rows)
        print(f"Filtered dataset: {len(filtered_df)} out of {len(self.df)} samples have corresponding images")

        return filtered_df
    
    def get_hidden_states(self, image_path):
        if self.embeddings_h5 is None:
            print(f"Debug: embeddings_h5 is None")
            return None

        # Extract image name from path
        image_filename = os.path.basename(image_path)
        image_name_base = os.path.splitext(image_filename)[0]
        image_name = str(image_name_base).strip()

        unwanted_strings=["head_","torso_","upper_ext_","lower_ext_"]

        for unwanted in unwanted_strings:
            if unwanted in image_name:
                image_name = image_name.replace(unwanted, "")


        # Check if we have embeddings for this image
        if 'embeddings' not in self.embeddings_h5:
            print(f"Debug: No 'embeddings' group found")
            return None

        embeddings_group = self.embeddings_h5['embeddings']

        # Try different possible keys
        possible_keys = [
            image_filename,  # Full filename with extension
            image_name,  # Base name without extension
            image_name + '.jpg',
            image_name + '.jpeg',
            image_name + '.png',
        ]

        img_group = None
        actual_key = None

        for key in possible_keys:
            if key in embeddings_group:
                img_group = embeddings_group[key]
                actual_key = key
                # print(f"Debug: Found embeddings with key: {key}")
                break

        if img_group is None:
            print(f"Debug: No embeddings found for any key: {possible_keys}")
            # Show some available keys for debugging
            available_keys = list(embeddings_group.keys())[:10]
            print(f"Debug: Available keys (first 10): {available_keys}")
            return None

        # Load hidden states from the specified layers
        hidden_states = []

        layer_idxs=[21,23,25,27,29,31,32]
        for layer_idx in range(self.layer_range[0], self.layer_range[1] + 1):
            if layer_idx in layer_idxs:
                layer_key = f'layer_{layer_idx}'
                if layer_key in img_group:
                    try:
                        # Load as tensor and squeeze the first dimension (batch dimension)
                        layer_data = img_group[layer_key][:]  # Shape: (1, 605, 4096)
                        layer_tensor = torch.from_numpy(layer_data).squeeze(0)  # Shape: (605, 4096)
                        # hidden_states[layer_idx] = layer_tensor
                        hidden_states.append(layer_tensor)
                        # print(f"Debug: Loaded {layer_key} with shape {layer_tensor.shape}")
                    except Exception as e:
                        print(f"Debug: Error loading {layer_key}: {e}")
            else:
                continue
    
        hidden_states=torch.stack(hidden_states)
        
        return hidden_states
    
    def __len__(self):
        return len(self.current_df)


    def __getitem__(self, idx):

        row = self.current_df.iloc[idx]

        image_name = str(row['image_name']).strip()

        possible_extensions = ['.jpeg', '.jpg', '.png', '']
        image_path = None

        for ext in possible_extensions:
            test_path = os.path.join(self.image_dir, image_name)
            if os.path.exists(test_path):
                image_path = test_path
                break
        if image_path is None:
            raise FileNotFoundError(f"Image not found for base name: {image_name}")

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        # Get patch indices
        patch_indices = row['indices']
        patch_indices = ast.literal_eval(patch_indices)

        # Prepare labels
        labels = {
            "trauma": row['trauma'],
        }

        # Get pre-computed hidden states
        hidden_states = self.get_hidden_states(image_path)
  
        L,S,D = hidden_states.shape
        pe_layer= sinusoidal_encoding(L,D).unsqueeze(1)
        
        hidden_states = hidden_states + pe_layer
        
        
        rest=[]
        for i in range(605):
            rest.append(i)
        
        rest=set(rest)-set(patch_indices)
        rest=list(rest)
        augment=hidden_states[:,rest,:]
        # print(hidden_states.shape)
        hidden_states=hidden_states[:,29:,:]                   ############################################### HARDCODED FOR PROMPT
        # print(hidden_states.shape)
        hidden_states=hidden_states[:,patch_indices,:]

        min_n,max_n=16,20
        N=np.random.randint(min_n,max_n)
        augment=augment.view(-1,augment.size(-1))
        indices=torch.randperm(augment.size(0))[:N]
        samples=augment[indices]
        hidden_states=hidden_states.float()
        L,S,D=hidden_states.shape
        # hidden_states=hidden_states.view(L*S,D)
        head = model.model.language_model.lm_head
        states = head(hidden_states)
        probab=torch.nn.Softmax(dim=2)
        out=probab(states)
        
        out1,out2=torch.max(out,2)
        # print(f"shape of out1 {out1.shape}")
        L,S=out1.shape
  
        # print("#"*10)
        # print(out1.shape)
        threshold = 0.1

        mask = out1 > threshold
    
        # print("out1",out1.shape)

        x = hidden_states[mask]
        
        aug_sequence = torch.cat([x,samples],dim=0)
        
        std = random.uniform(0.1,0.9)
        noise = torch.randn_like(aug_sequence) * std
        
        aug_sequence = aug_sequence + noise
       

        return image_tensor, labels, x, image_name
    
def sinusoidal_encoding(length, dim):
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length,dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float()*(-math.log(10000.0)/dim))
    pe[:, 0::2] =torch.sin(position*div_term)
    pe[:, 1::2] =torch.cos(position*div_term)
  
    return pe                    #[L,D]

def pad_sequence_length_only(tensors):
    # tensors: list of [S_i, D]
    max_seq_len = []

    att_mask=[]
    padded = []
    for t in tensors:
        max_seq_len.append(t.shape[0])
    # print(max_seq_len)
    max_size=max(max_seq_len)
    # print("pad_len::",pad_len)
    for t in tensors:
        S_i, D = t.shape
        
        pad_len = max_size - t.shape[0]
        if pad_len > 0:
            # Pad on dimension 1 (sequence length)
            t_padded = F.pad(t, (0, 0, 0, pad_len))
            # print("after_pad::",t_padded.shape)  # Pad format: (last_dim..., seq_len)
        else:
            t_padded = t
        padded.append(t_padded)

        mask = torch.ones((S_i), dtype=torch.float32)
        # print("mask::",mask.shape)
        if pad_len > 0:
            pad_mask = F.pad(mask, (0, pad_len), value=0)
        else:
            pad_mask = mask[:max_size]
            
        # print("pad mask::", pad_mask.shape)

        att_mask.append(pad_mask)

    padded_tensor = torch.stack(padded, dim=0)          # [B, max_S, D]
    attention_mask = torch.stack(att_mask, dim=0)       # [B, max_S]

    return padded_tensor, attention_mask

def pad_sequence_length_only_2(tensors):
    # tensors: list of [S_i, D]
    max_seq_len = 2000

    att_mask=[]
    padded = []
  
    for t in tensors:
        S_i, D = t.shape
        
        pad_len = max_seq_len - t.shape[0]
        mask = torch.ones((S_i), dtype=torch.float32)

        if pad_len > 0:
            # Pad on dimension 1 (sequence length)
            t_padded = F.pad(t, (0, 0, 0, pad_len))
            pad_mask = F.pad(mask, (0, pad_len), value=0)
            # print("after_pad::",t_padded.shape)  # Pad format: (last_dim..., seq_len)
        else:
            t_padded = t
            pad_mask = mask[:max_seq_len]

        padded.append(t_padded)
        att_mask.append(pad_mask)

        # print("mask::",mask.shape)            
        # print("pad mask::", pad_mask.shape)


    padded_tensor = torch.stack(padded, dim=0)          # [B, max_S, D]
    attention_mask = torch.stack(att_mask, dim=0)       # [B, max_S]
   
    return padded_tensor, attention_mask

def custom_collate_fxn(batch):
    images, label_dicts, hidden_states, image_names = zip(*batch)

    images = torch.stack(images, dim=0)
    # print(hidden_states[0].shape)
    # Extract labels
    label_vals = [int(ld.get("trauma", "-1")) for ld in label_dicts]
    labels = torch.tensor(label_vals, dtype=torch.long)
    # print(f"labels:{labels}")
    batch_cls_embeds,mask=pad_sequence_length_only_2(hidden_states)
    # print("hidden_states::",batch_cls_embeds.shape)
    
    # [B, L×S, D]
    B, S, D = batch_cls_embeds.shape
    # batch_cls_embeds = batch_cls_embeds.view(B,L * S, D)
    # print(batch_cls_embeds.shape)
    B,S= mask.shape

    return images, labels, batch_cls_embeds, mask, image_names

if __name__ == "__main__":

    img_dir = "/home/uas-dtu/Desktop/segmented_images/train_data"
    csv_path = "/home/uas-dtu/llava-interp-main/sorted_csv.csv"
    embeddings_file = "/media/uas-dtu/e117d1c8-5ec6-446b-8141-37a5493e1fb2/FULL_precomputed_embeddings.h5"

    # Create dataset
    dataset = MLLMDataset(img_dir, csv_path, embeddings_file)


    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=custom_collate_fxn,
        num_workers=16
    )

    for batch_idx, (images, labels_batch, indices_batch, hidden_batch, image_names) in enumerate(dataloader):
        print(labels_batch,hidden_batch.shape,image_names)

        # if batch_idx >= 0:
        #     break
        continue