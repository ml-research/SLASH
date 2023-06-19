# Select the base image
#old 21.06-py3 
#FROM nvcr.io/nvidia/pytorch:21.12-py3
FROM nvcr.io/nvidia/pytorch:21.06-py3


# Select the working directory
WORKDIR  /SLASH

# Install PyTorch
#RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install Python requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install --ignore-installed -r requirements.txt
# RUN conda install -c potassco clingo=5.5.0
# RUN pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git ; pip install rtpt torchsummary ; python -m pip install -U scikit-image
# RUN conda install -c conda-forge tqdm
# RUN conda install -c anaconda seaborn
# RUN conda install -c conda-forge scikit-learn
# RUN conda install -c conda-forge tensorboard   

# Remove fontlist file so MPL can find the serif fonts
RUN rm -rf /.matplotlib/fontlist-v330.json

# Setup mpl config dir since SciencePlots install .mplstyle files into this dir
RUN mkdir -p /.matplotlib/stylelib
RUN chmod a+rwx -R /.matplotlib 
ENV MPLCONFIGDIR=/.matplotlib

# Add fonts for serif rendering in MPL plots
RUN apt-get update
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt-get install --yes ttf-mscorefonts-installer
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime
RUN apt-get install dvipng cm-super fonts-cmu --yes
RUN apt-get install fonts-dejavu-core --yes
RUN apt install -y tmux
# RUN fc-cache -f -v 

RUN python3 -m pip install setuptools==59.5.0
RUN pip install pandas==1.3.0