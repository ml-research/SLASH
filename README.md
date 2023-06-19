#  Scalable Neural-Probabilistic Answer Set Programming

Arseny Skryagin, Wolfgang Stammer, Daniel Ochs, Devendra Singh Dhami , Kristian Kersting  
 
<p align="center">
  <img src="./imgs/slash_icon.png">
</p>

[![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Abstract
The goal of combining the robustness of neural networks and the expressiveness of symbolic
methods has rekindled the interest in Neuro-Symbolic AI. Deep Probabilistic Programming
Languages (DPPLs) have been developed for probabilistic logic programming to be carried
out via the probability estimations of deep neural networks. However, recent SOTA DPPL
approaches allow only for limited conditional probabilistic queries and do not offer the power
of true joint probability estimation. In our work, we propose an easy integration of tractable
probabilistic inference within a DPPL. To this end, we introduce SLASH, a novel DPPL
that consists of Neural-Probabilistic Predicates (NPPs) and a logic program, united via
answer set programming (ASP). NPPs are a novel design principle allowing for combining
all deep model types and combinations thereof to be represented as a single probabilistic
predicate. In this context, we introduce a novel +/− notation for answering various types
of probabilistic queries by adjusting the atom notations of a predicate. To scale well, we
show how to prune the stochastically insignificant parts of the (ground) program, speeding
up reasoning without sacrificing the predictive performance. We evaluate SLASH on a
variety of different tasks, including the benchmark task of MNIST addition and Visual
Question Answering (VQA).




## Introduction
This is the repository for SLASH, the deep declarative probabilistic programming language introduced within **Neural-Probabilistic Answer Set Programming** [![KR](https://img.shields.io/badge/Conference-KR2022-blue)](https://kr2022.cs.tu-dortmund.de/index.php) and **Scalable Neural-Probabilistic Answer Set Programming** [![arXiv](https://img.shields.io/badge/arXiv-2306.08397-<COLOR>.svg)](https://arxiv.org/abs/2306.08397). [Link](https://arxiv.org/abs/2306.08397) to the preprint.
 


## 1. Prerequisites
### 1.1 Cloning SLASH and submodules
To clone SLASH including all submodules use:
```
git clone --recurse-submodules -j8 https://github.com/ml-research/SLASH
```

### 1.2 Anaconda
The `environment.yaml` provides all packages needed to run a SLASH program using the Anaconda package manager. To create a new conda environment install Anaconda  ([installation page](https://docs.anaconda.com/anaconda/install/)) and then create a new environment using the following command:
```
conda env create -n slash -f environment.yml
conda activate env_dev
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
```

The following packages are installed:
- pytorch
- clingo  # version 5.5.1
- scikit-image
- scikit-learn 
- seaborn
- tqdm
- rtpt
- tensorboard  # needed for standalone slot attention
- torchsummary
- GradualWarmupSheduler

### 1.3 Virtual environment
To use virtualenv run the following commands. This creates a new virtual environment and installs all packages needed. Tested Python Versions 3.6. For using cuda version `10.x` with pytorch remove the `+cu113` version appendix in the `requirements.txt` file otherwise it will use CUDA version `11.3`.
```
python3 -m venv slash_env
source slash_env/bin/activate
pip install --upgrade pip
python3 -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

### 1.4 Docker
Alternatively you can also run slash using docker. For this you first need to create an image using the provided `Dockerfile`.

To start create an image named `slash` and then run the container using that image.  In the slash base folder execute: 
```
docker build . -t slash:0.01
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=<GPU-ID> --ipc=host -it --rm  -v /$(pwd):/slash slash:1.0
```


## 2. Project Structure
```
--data: contains datasets after downloading
--src: source files which has its own readme. Go there for more details.
```


## 3. Getting started
Go visit the  ``` src/experiments/mnist_top_k``` folder to get familar with SLASH and run one of the training scripts.    


## Citation

If you use this code for your research, please cite our conference paper at [KR22](https://proceedings.kr.org/2022/48/)

```
@inproceedings{skryagin2022KR,
  title={Neural-Probabilistic Answer Set Programming},
  author={Arseny Skryagin and Wolfgang Stammer and Daniel Ochs and Devendra Singh Dhami and Kristian Kersting},
  booktitle={Proceedings of the 19th International Conference on Principles of Knowledge Representation and Reasoning (KR)},
  year={2022}
}
```

## Acknowlegments  
This work was partly supported by the Federal Minister of Education and Research (BMBF) and the Hessian Ministry of Science and the Arts (HMWK) within the National Research Center for Applied Cybersecurity ATHENE,  the ICT-48 Network of AI Research Excellence Center "TAILOR" (EU Horizon 2020, GA No 952215, and the Collaboration Lab with Nexplore "AI in Construction" (AICO). It also benefited from the BMBF AI lighthouse project, the Hessian research priority programme LOEWE within the project WhiteBox, the HMWK cluster projects "The Third Wave of AI" and "The Adaptive Mind", the German Center for Artificial Intelligence (DFKI) project "SAINT".