import torch
import torch.nn as nn

class ReduLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_classes = 0
    
    def __name__(self):
        return 'ReduNet'
    
    def forward(self, Z):
        raise NotImplementedError
    
    def zero(self):
        state_dict = self.state_dict()
        state_dict['E.weight'] = torch.zeros_like(self.E.weight)
        for i in range(self.num_classes):
            state_dict[f'Cs.{i}.weight'] = torch.zeros_like(self.Cs[i].weight)
        self.load_state_dict(state_dict)

    def compute_gam(self, X, y):
        pass

    def compute_E(self, X: torch.Tensor) -> torch.Tensor:
        pass

    def compute_Cs(self, X, y):
        pass

    def init(self, X, y):
        gam = self.compute_gam(X, y)
        E = self.compute_E(X)
        Cs = self.compute_Cs(X, y)
        self.set_params(E, Cs, gam)

    def update_old(self, X: torch.Tensor, y, tau):
        E = self.compute_E(X).to(X.device)
        Cs = self.compute_Cs(X, y)
        state_dict = self.state_dict()
        ref_E = self.E.weight
        ref_Cs = [self.Cs[i].weight for i in range(self.num_classes)]
        new_E = ref_E + tau * (E - ref_E)
        new_Cs = [ref_Cs[j] + tau * (Cs[j] - ref_Cs[j]) for j in range(self.num_classes)]
        state_dict['E.weight'] = new_E
        for i in range(self.num_classes):
            state_dict[f'Cs.{i}.weight'] = new_Cs[i]
        self.load_state_dict(state_dict)

    def set_params(self, E: torch.Tensor, Cs: torch.Tensor, gam: torch.Tensor = None):
        state_dict = self.state_dict()
        assert self.E.weight.shape == E.shape, f'E shape does not match: {self.E.weight.shape} and {E.shape}'
        state_dict['E.weight'] = E
        for i in range(self.num_classes):
            assert self.Cs[i].weight.shape == Cs[i].shape, f'Cj shape does not match'
            state_dict[f'Cs.{i}.weight'] = torch.zeros_like(self.Cs[i].weight)
        if gam is not None:
            assert self.gam.shape == gam.shape, 'gam shape does not match'
            state_dict['gam'] = gam
        self.load_state_dict(state_dict)
    
    def get_params(self):
        E = self.E.weight
        Cs = [self.Cs[j].weight for j in range(self.num_classes)]
        return E, Cs