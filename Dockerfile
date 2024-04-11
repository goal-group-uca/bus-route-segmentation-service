FROM --platform=linux/amd64  ubuntu:22.04
# Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

## Install miniconda
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm -rf /tmp/*


ADD ./eMob_Barcelona/ ./eMob_Barcelona/
ADD ./routes/ ./eMob_Barcelona/src/middle_output_perfect/

RUN conda config --append channels conda-forge
RUN conda env create --file ./eMob_Barcelona/gdal.yml 

RUN conda init bash
RUN echo "conda run -n gdal" > ~/.bashrc
SHELL ["conda", "run", "-n", "gdal", "/bin/bash", "-c"]
RUN pip install -r eMob_Barcelona/requirements.txt
SHELL ["conda", "run", "-n", "gdal", "/bin/bash", "-c"]
RUN pip install geopandas
SHELL ["conda", "run", "-n", "gdal", "/bin/bash", "-c"]
RUN pip install bottle

WORKDIR "./eMob_Barcelona/"
ENV GEOAPIFY_KEY "your_api_key"
ENV PATH /opt/conda/envs/gdal/bin:$PATH
RUN /bin/bash -c "source activate gdal"

ENTRYPOINT [ "python3", "web_app.py"]
