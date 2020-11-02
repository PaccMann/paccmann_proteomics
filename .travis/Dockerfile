FROM drugilsberg/rdkit-ubuntu:latest
RUN apt-get update && apt-get install -y git
WORKDIR /paccmann_proteomics
# install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# copy paccmann_proteomics
COPY . .
RUN pip3 install --no-deps .
CMD /bin/bash