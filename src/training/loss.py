# Loss functions to compare the latents
import torch
from torch import nn


def get_positive_sum(model_pred_emb, temperature):
    # Flatten the embeddings
    model_pred_emb = model_pred_emb.reshape(model_pred_emb.shape[0], -1)

    # Normalize
    model_pred_emb = model_pred_emb / (model_pred_emb.norm(p=2, dim=1, keepdim=True) + 1e-20)

    # Calculate similarity values
    token_sim = torch.mm(model_pred_emb, model_pred_emb.T)

    token_sim_exp = torch.exp(token_sim / temperature)

    # Mask for positive samples
    num_samples = model_pred_emb.shape[0]
    # Calculate for different latent with same directions - upper triangle to not get repeating similarities
    mask = (torch.ones((num_samples, num_samples)).triu() * (1 - torch.eye(num_samples))).to(model_pred_emb.device).bool()
    # tensor([[False,  True],
    #     [False, False]], device='cuda:0')
    
    pos_sum = token_sim_exp.masked_select(mask).sum()
    pos_count = mask.sum()

    return pos_sum, pos_count, token_sim, token_sim_exp


def get_negative_sum(model_pred_emb, model_pred_content, temperature):

    batch_size = model_pred_emb.shape[0]
    model_pred_emb = model_pred_emb.reshape(batch_size, -1)
    model_pred_content = model_pred_content.reshape(batch_size, -1)

    # Normalize 
    model_pred_emb = model_pred_emb / (model_pred_emb.norm(p=2, dim=1, keepdim=True) + 1e-20)
    model_pred_content = model_pred_content / (model_pred_content.norm(p=2, dim=1, keepdim=True) + 1e-20)

    # Calculate the similarity values
    token_sim = torch.mm(model_pred_emb, model_pred_content.T)

    token_sim_exp = torch.exp(token_sim / temperature)

    # Build a mask with zeros on the diagonal and ones elsewhere.
    mask = (~torch.eye(batch_size, dtype=torch.bool, device=model_pred_emb.device))
    # [[False,  True,  True],
    #  [ True, False,  True],
    #  [ True,  True, False]]

    # Select negative samples with the mask.
    neg_sum = token_sim_exp.masked_select(mask).sum()
    neg_count = mask.sum()

    return neg_sum, neg_count, token_sim, token_sim_exp


def get_positive_sim(model_pred_emb):
    # Flatten the embeddings
    model_pred_emb = model_pred_emb.reshape(model_pred_emb.shape[0], -1)

    # Normalize
    model_pred_emb = model_pred_emb / (model_pred_emb.norm(p=2, dim=1, keepdim=True) + 1e-20)

    # Calculate similarity values
    token_sim = torch.mm(model_pred_emb, model_pred_emb.T)

    # Mask for positive samples
    num_samples = model_pred_emb.shape[0]
    # Calculate for different latent with same directions - upper triangle to not get repeating similarities
    mask = (torch.ones((num_samples, num_samples)).triu() * (1 - torch.eye(num_samples))).to(model_pred_emb.device).bool()
    # tensor([[False,  True],
    #     [False, False]], device='cuda:0')
    
    pos_sum = token_sim.masked_select(mask).sum()
    pos_count = mask.sum()

    return pos_sum, pos_count, token_sim
