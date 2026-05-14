import torch
import torch.nn as nn
from typing import Union


class EmbedAtom(nn.Module):
    def __init__(self, embed_dim: int, max_atom_type: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_atom_type = max_atom_type
        # linear after onehot
        self.lin = nn.Linear(self.max_atom_type, self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, atom_type: torch.Tensor) -> torch.Tensor:
        # atom_type: (num_nodes,)
        # onehot: (num_nodes, max_atom_type)
        onehot = nn.functional.one_hot(
            atom_type, num_classes=self.max_atom_type
        ).float()
        # embeded: (num_nodes, embed_dim)
        embeded = self.lin(onehot)
        embeded = self.norm(embeded)
        return embeded


class EmbedDist(nn.Module):
    def __init__(self, embed_func: str, embed_dim: int, cutoff: float):
        super().__init__()
        self.embed_func = embed_func
        self.embed_dim = embed_dim
        self.cutoff = cutoff
        self.lin = nn.Linear(self.embed_dim, self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)

    def gaussian_emb(
        self,
        dist: torch.Tensor,
        start: Union[float, None] = None,
        end: Union[float, None] = None,
    ):
        if start is None:
            start = 0.0
        if end is None:
            end = self.cutoff
        # grid: (embed_dim,)
        grid = torch.linspace(start, end, self.embed_dim, device=dist.device)
        # gaussian: (num_edges, embed_dim)
        gaussian = torch.exp(-(((dist.unsqueeze(-1) - grid) / self.cutoff) ** 2))
        # embeded: (num_edges, embed_dim)
        embeded = self.lin(gaussian)
        embeded = self.norm(embeded)
        return embeded

    def bessel_emb(self, dist: torch.Tensor):
        pass

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: (num_edges,)
        if self.embed_func == "gaussian":
            return self.gaussian_emb(dist)
        else:
            raise NotImplementedError(f"Not implemented yet: {self.embed_func}")

class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        dist_emb_func: str,
        # atom_emb_func: str,
        embed_dim: int,
        max_atom_type: int,
        cutoff: float,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_atom = EmbedAtom(embed_dim, max_atom_type)
        self.embed_dist = EmbedDist(dist_emb_func, embed_dim, cutoff)

    def forward(self, atom_type: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        # atom_type: (num_nodes,)
        # dist: (num_edges,)
        # embed_atom: (num_nodes, embed_dim)
        embed_atom = self.embed_atom(atom_type)
        # embed_dist: (num_edges, embed_dim)
        embed_dist = self.embed_dist(dist)
        return embed_atom, embed_dist
