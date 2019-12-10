# dockerfile install for ACVProject
# Tim Burt, Yuan Zi

# We will use Ubuntu for our image
FROM ubuntu

# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade

# Adding wget and bzip2

RUN apt-get install -y wget bzip2

# Miniconda installing
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

RUN eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
RUN bash -i Miniconda3-latest-Linux-x86_64.sh -b
RUN rm Miniconda3-latest-Linux-x86_64.sh

# Set path to miniconda
ENV PATH /root/miniconda3/bin:$PATH

RUN eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
SHELL ["/bin/bash", "-c"]

RUN conda create --name ACVProject

RUN apt-get install python3-pip -y
RUN apt-get install git -y

RUN git config --global url.https://a61b25f11e21c6d54aafce7c01257fe06cb01aa3@github.com/.insteadOf https://github.com/

RUN git clone https://github.com/lpopov101/ACVProject.git


# Updating miniconda packages
RUN pip install PyQt5
RUN conda install -c conda-forge tensorflow
RUN conda install -c conda-forge keras
RUN conda install -c anaconda numpy
RUN conda install -c anaconda scikit-image
RUN conda install -c conda-forge opencv
RUN conda install -c anaconda scipy
RUN conda install -c conda-forge pydicom
RUN conda install -c conda-forge matplotlib
RUN conda install -c anaconda scikit-learn

RUN apt-get install unzip -y

# Configuring access to Jupyter
#RUN mkdir /opt/notebooks
#RUN jupyter notebook --generate-config --allow-root
#RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py
# Jupyter listens port: 8888
#EXPOSE 8888
# Run Jupytewr notebook as Docker main process
#CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/opt/notebooks", "--ip='*'", "--port=8888", "--no-browser"]