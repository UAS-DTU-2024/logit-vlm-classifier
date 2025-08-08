import torch
import torch.nn as nn
from src.HookedLVLM import HookedLVLM


device ="cuda"
class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4.0, dropout_ratio=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, x, attention_mask=None):
        

        residual = x
        x = self.norm1(x)

        if attention_mask is not None:
            key_padding_mask = attention_mask.bool()
            attn_out, attn_weights = self.attn(x, x, x, key_padding_mask=key_padding_mask,need_weights=True)
        else:
            attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        # print(f"attn_out:{attn_out.shape}")
        # print(f"attn_wegihts:{attn_weights.shape}")

        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, attn_weights
    
class mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
        )
    
    def forward(self, x):
        x = self.ffn(x)
        return x
    
class projector(nn.Module):
    def __init__(self, dim, dropout_ratio=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(2048, dim),
            nn.Dropout(dropout_ratio)
        )
    
    def forward(self, x):
        x = self.ffn(x)
        return x




class encoder_lmhead(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, num_blocks=1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 4096))
        model = HookedLVLM(device='cpu',quantize=False)    #device not set for balanced mode
        self.blocks = nn.Sequential(
            *[Encoder(dim, num_heads, mlp_ratio) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = model.model.language_model.lm_head.to(device)
        for param in self.head.parameters():
            param.requires_grad = False 


    def forward(self, x, attention_mask=None, output_attentions=False):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        for block in self.blocks:
            x, attention_weights = block(x, attention_mask=attention_mask)
        x = self.norm(x)
        cls_token = x[:,0,:]
        cls_logits = self.head(cls_token)
        if output_attentions:
            return cls_logits, x,  attention_weights
        else:
            return cls_logits, x


class reduced_edim_moreblocks(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, num_blocks=4, num_classes=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 4096))
        model = HookedLVLM(device='cpu',quantize=False)
        self.blocks = nn.Sequential(
            *[Encoder(1024, num_heads, mlp_ratio) for _ in range(num_blocks)]
        )
        self.ffn = mlp(dim, dropout_ratio=0.1)
        self.reffn = projector(dim, dropout_ratio=0.1)
      
        self.norm = nn.LayerNorm(1024)
        self.head = model.model.language_model.lm_head.to(device)
        for param in self.head.parameters():
            param.requires_grad = False 
        
    
    def forward(self, x, attention_mask=None, output_attentions=False):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = self.ffn(x)

        for block in self.blocks:
            x, attention_weights = block(x, attention_mask=attention_mask)
        
        x = self.norm(x)
        cls_token = x[:, 0, :] 
        reproject = self.reffn(cls_token)

        cls_logits = self.head(reproject) 


        if output_attentions:
            return cls_logits, attention_weights
        else:
            return cls_logits

class reduced_classifier_moreblocks(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, num_blocks=4, num_classes=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 4096))

        self.blocks = nn.Sequential(
            *[Encoder(512, num_heads, mlp_ratio) for _ in range(num_blocks)]
        )
        self.ffn = mlp(dim)
        self.classify = nn.Linear(512, num_classes)
        self.norm = nn.LayerNorm(512)
    
    def forward(self, x, attention_mask=None, output_attentions=False):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = self.ffn(x)

        for block in self.blocks:
            x, attention_weights = block(x, attention_mask=attention_mask)
        
        x = self.norm(x)
        cls_token = x[:, 0, :] 
        
        output_class= self.classify(cls_token)    


        if output_attentions:
            return output_class, attention_weights
        else:
            return output_class