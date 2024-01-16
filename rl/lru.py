# Linear Recurrent Units (LRUs)
# Introduced in:
# "Resurrecting Recurrent Neural Networks for Long Sequences" (Orvieto 2023)
# https://arxiv.org/pdf/2303.06349.pdf

# From https://github.com/adrian-valente/lru_experiments/
# Another good example here: https://github.com/bojone/rnn/blob/main/lru.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LRU(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_hidden: int,
                 d_out: int,
                 r_min: float = 0.,
                 r_max: float = 1.,
                 max_phase: float = 6.28
                 ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

        self.theta_log = nn.Parameter(torch.empty(d_hidden))
        self.nu_log = nn.Parameter(torch.empty(d_hidden))
        self.gamma_log = nn.Parameter(torch.empty(d_hidden))
        self.B_re = nn.Parameter(torch.empty(d_hidden, d_in))
        self.B_im = nn.Parameter(torch.empty(d_hidden, d_in))
        self.C_re = nn.Parameter(torch.empty(d_out, d_hidden))
        self.C_im = nn.Parameter(torch.empty(d_out, d_hidden))
        self.D = nn.Parameter(torch.empty(d_out, d_in))

        self._init_params()

    def diag_lambda(self) -> torch.Tensor:
        return torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))

    def _init_params(self):
        nn.init.uniform_(self.theta_log, a=0, b=self.max_phase)

        u = torch.rand((self.d_hidden,))
        nu_init = torch.log(-0.5 * torch.log(u * (self.r_max**2 - self.r_min**2) + self.r_min**2))
        with torch.no_grad():
            self.nu_log.copy_(nu_init)
            diag_lambda = self.diag_lambda()
            self.gamma_log.copy_(torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2)))

        nn.init.xavier_normal_(self.B_re)
        nn.init.xavier_normal_(self.B_im)
        nn.init.xavier_normal_(self.C_re)
        nn.init.xavier_normal_(self.C_im)
        nn.init.xavier_normal_(self.D)  # Set something like diagonal matrix eventually

    def forward(self, u: torch.Tensor, init_states: torch.Tensor = None) -> torch.Tensor:
        diag_lambda = self.diag_lambda()
        B_norm = torch.diag(self.gamma_log).to(torch.cfloat) @ (self.B_re + 1j * self.B_im)
        C = self.C_re + 1j * self.C_im

        # Initial states can be a vector of shape (d_hidden,) or a matrix of shape (batch_size, d_hidden)
        if init_states is not None and init_states.ndim == 1:
            init_states = init_states.unsqueeze(0)

        h = init_states.to(torch.cfloat) if init_states is not None \
                else torch.zeros((u.shape[0], self.d_hidden), dtype=torch.cfloat, device=self.theta_log.device)
        outputs = []
        # FIXME: Incorporate https://github.com/proger/accelerated-scan
        for t in range(u.shape[1]):
            h = h * diag_lambda + u[:, t].to(torch.cfloat) @ B_norm.T
            y = torch.real(h @ C.T) + u[:, t] @ self.D.T
            outputs.append(y)

        return torch.stack(outputs, dim=1)

    def __repr__(self):
        return f"LRU(d_in={self.d_in}, d_hidden={self.d_hidden}, d_out={self.d_out})"
