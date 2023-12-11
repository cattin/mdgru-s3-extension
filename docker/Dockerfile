FROM tensorflow/tensorflow:1.15.0-gpu

# Remove source list to fix missing pub key from NVIDIA package
RUN rm /etc/apt/sources.list.d/*
# Must have packages
RUN apt-get update && apt-get install -y git python3-pip
RUN apt-get upgrade -y --allow-unauthenticated

RUN pip3 install --upgrade pip==20.3.1
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV SSL_CERT_FILE=/app/docker/CAs/dbe-object-storage-ca.crt
ENV AWS_CA_BUNDLE=/app/docker/CAs/dbe-object-storage-ca.crt
WORKDIR  /app

ENTRYPOINT ["python3", "RUN_mdgru.py"]
