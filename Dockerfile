FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    foscat \
    healpy \
    xarray \
    zarr \
    dask \
    s3fs \
    numpy \
    matplotlib \
    pint-xarray \
    cf-xarray \
    cmcrameri

WORKDIR /app
COPY . /app

CMD ["python", "01_sst_gap_filling.py"]
