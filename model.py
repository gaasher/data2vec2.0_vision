import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from x_transformers import Encoder

'''
PatchEmbed class, adapted from https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632 I think, but I dont have medium premium so idk
- This class is used to convert the image into patches using a convolutional layer
'''
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=64):
        super().__init__()
        if isinstance(img_size, int):
            img_size = img_size, img_size
        if isinstance(patch_size, int):
            patch_size = patch_size, patch_size
        #calculate the number of patches
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        #convolutional layer to convert the image into patches
        self.conv = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        

    def forward(self, x):
        x = self.conv(x)
        #flatten the patches
        x = rearrange(x, 'b e h w -> b (h w) e')
        return x
'''
Decoder class:
- takes in a recoonstruction of the masked and unmasked tokens/patches and returns a prediction of the target embedding (same size as the input)
- This uses 2d convolutions to reconstruct the target representation since I implemented the vision data2vec model, but 1d cnn's can be used for text data2vec
- Uses a GELU activation function and layer normalization
'''
class Decoder(nn.Module):
    def __init__(self,
                 depth,
                 num_tokens,
                 embed_dim,
                 decoder_dim,
                 kernel_size,
                 padding,
                 groups,
                 ):
        super().__init__()
        #back calculate the height and width of the image
        self.h, self.w = int(math.sqrt(num_tokens)), int(math.sqrt(num_tokens))
        
        #create a list of layers
        self.convs = nn.ModuleList()
        
        #add the first layer, converting to the decoder dimension (b x embed_dim x h x w -> b x decoder_dim x h x w)
        self.convs.append(nn.Conv2d(embed_dim, decoder_dim, kernel_size=kernel_size, padding=padding, groups=groups))
        self.convs.append(nn.LayerNorm((decoder_dim, self.h, self.w)))
        self.convs.append(nn.GELU())
        
        #add the remaining layers
        for i in range(depth - 1):
            self.convs.append(nn.Conv2d(decoder_dim, decoder_dim, kernel_size=kernel_size, padding=padding, groups=groups))
            self.convs.append(nn.LayerNorm((decoder_dim, self.h, self.w)))
            self.convs.append(nn.GELU())
        
        #project back to the embedding dimension
        self.proj = nn.Linear(decoder_dim, embed_dim)
        
    def forward(self, x):
        #reshape the input to the correct dimensions, as implemented in the fairseq code
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.h, w=self.w)
        
        #use a residual connection
        residual = x
        
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i > 0:
                x = x + residual
            residual = x
        
        #project back to the embedding dimension
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj(x)
        return x

'''
data2vec_base class:
- takes in batch of images and returns a prediction of the target embedding if is_teacher is False, otherwise returns the target embedding
- This class is used to implement the vision data2vec model, but can be adapted for text data2vec by changing the patch embedding and decoder
- adapted in many ways from lucidrains' implementation of MAE, and uses x-transformers, a library by the same author for the encoder
'''
class data2vec_base(nn.Module):
    def __init__(
            self,
            img_size=224, #img_size is the height and width of the image
            patch_size=16, #patch_size is the height and width of the patches
            in_chans=3, #in_chans is the number of channels in the image
            embed_dim=64, #embed_dim is the embedding dimension of the patches/tokens
            masking_ratio=0.5, #masking_ratio is the ratio of tokens to mask
            heads=8, #heads is the number of attention heads in the encoder
            depth=8, #depth is the number of layers in the encoder
            decoder_depth=3, #decoder_depth is the number of layers in the decoder
            decoder_dim=64, #decoder_dim is the dimension of the decoder
            decoder_kernel_size=3, #decoder_kernel_size is the kernel size of the decoder
            decoder_padding=1, #decoder_padding is the padding of the decoder
            decoder_groups=1, #decoder_groups is the number of groups in the decoder
            post_emb_norm=True, #post_emb_norm is whether to apply layer normalization after the embedding
            dropout=0., #dropout is the dropout rate
            is_teacher = False, #is_teacher is whether the model is being used as a teacher
            k=4, #k is the number of transformer blocks to use as the teacher's 'context' for target creation
    ):
        super().__init__()
        if is_teacher:
            assert(k > 0 and masking_ratio == 0)
        
        #define the parameters
        self.masking_ratio = masking_ratio
        self.is_teacher = is_teacher
        self.k = k
        self.num_tokens = int(img_size**2/patch_size**2)
        
        #define the patch embedding and positional embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens+1, embed_dim))
        
        #define the cls and mask tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, 0.02)
        nn.init.trunc_normal_(self.mask_token, 0.02)

        #define the encoder and decoder, as well as the layer normalization and dropout
        self.post_emb_norm = nn.LayerNorm(embed_dim) if post_emb_norm else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(
            dim=embed_dim,
            heads=heads,
            depth=depth, 
        )    
        self.decoder = Decoder(depth=decoder_depth, num_tokens=self.num_tokens, embed_dim=embed_dim, decoder_dim=decoder_dim, kernel_size=decoder_kernel_size, padding=decoder_padding, groups=decoder_groups)
    
    #generate the targets for the teacher model
    @torch.no_grad()
    def generate_targets(self, x:torch.Tensor, encoder: nn.Module, k:torch.Tensor=4):
        encoder = encoder.eval() #not sure if this is necessary
        
        _, intermediates = encoder(x, return_hiddens=True) #get intermediates from the encoder
        
        intermediates = torch.stack([h.clone() for h in intermediates.hiddens]) #extract the hidden states
        intermediates = intermediates[-k:] #top k hidden states
        b, n, h, w = intermediates.shape
        
        #normalize the hidden states
        intermediates = rearrange(intermediates, 'b n h w -> (b n) h w')     
        intermediates = F.instance_norm(intermediates)
        intermediates = rearrange(intermediates, '(b n) h w -> b n h w', b=b, n=n)
        
        intermediates = intermediates.mean(0)
        return intermediates[:, 1:] #return non cls token
    
    def forward(self, x):
        #get the patch embeddings
        x = self.patch_embed(x)
        b, n, e = x.shape
        
        #add positional embedding
        x = x + self.pos_embedding[:, :n]
        
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked - adapted from lucidrains' implementation of MAE
        num_masked = int(self.masking_ratio * n)
        rand_indices = torch.rand(b, n).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        # get the unmasked tokens to be encoded

        batch_range = torch.arange(b)[:, None]
        x = x[batch_range, unmasked_indices]
        #add cls and embedding tokens
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        #post embedding norm
        x = self.post_emb_norm(x)
        
        #get target representations if teacher
        if self.is_teacher:
            return self.generate_targets(x, self.encoder,self.k)
        
        #x shape: (b, n + 1, e)
        
        #encode the tokens
        x = self.dropout(x)
        x = self.encoder(x) 
        x = self.norm(x)

        #reconstruct the tokenss
        reconstruced = torch.zeros(b, n, e, dtype=x.dtype)
        cls_embedding = x[:, 0]
        reconstruced[batch_range, unmasked_indices] = x[:, 1:]
        reconstruced[batch_range, masked_indices] = self.mask_token 
        reconstruced.type_as(x)
        #reconstructed shape: (b, n, e)

        #decode the tokens
        decoded = self.decoder(reconstruced)
        
        return decoded