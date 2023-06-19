import torch
import torch.nn as nn

class Net_nn(nn.Module):
    def __init__(self):
        super(Net_nn, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # 6 is the output chanel size; 5 is the kernal size; 1 (chanel) 28 28 -> 6 24 24
            nn.MaxPool2d(2, 2),  # kernal size 2; stride size 2; 6 24 24 -> 6 12 12
            nn.ReLU(True),       # inplace=True means that it will modify the input directly thus save memory
            nn.Conv2d(6, 16, 5), # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True) 
        )
        self.classifier =  nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.Softmax(1)
        )

    def forward(self, x, marg_idx=None, type=1):
        
        assert type == 1, "only posterior computations are available for this network"

        # If the list of the pixel numbers to be marginalised is given,
        # then genarate a marginalisation mask from it and apply to the
        # tensor 'x'
        if marg_idx:
            batch_size = x.shape[0]
            with torch.no_grad():
                marg_mask = torch.ones_like(x, device=x.device).reshape(batch_size, 1, -1)
                marg_mask[:, :, marg_idx] = 0
                marg_mask = marg_mask.reshape_as(x)
                marg_mask.requires_grad_(False)
            x = torch.einsum('ijkl,ijkl->ijkl', x, marg_mask)
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x
