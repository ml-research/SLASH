# Source folder quick overview:   

1. `EinsumNetwork` contains code which was cloned from the [git repository EinNets](https://github.com/cambridge-mlg/EinsumNetworks).
2. `SLASH` contains all modifications made to enable probabilisitc circuits in the program, VQA and SLASH-attention, optimized gradient computation and the SAME technique.
    * `mvpp.py` contains all functions to compute stable models given a logic program and to compute the gradients using the output probabilites  
    * `slash.py` contains the SLASH class which brings together the symbolic program with neural nets or probabilistic circuits
3. `experiments` contains the VQA, mnist addtion and slash attention experiments