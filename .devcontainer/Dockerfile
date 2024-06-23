FROM python:3.12.1-slim-bullseye
USER root

RUN groupadd jovyan \
    && usermod -g jovyan -a -G users jovyan

USER jovyan

RUN apt-get update && \
    apt-get -y install --reinstall ca-certificates && \
    apt-get -y install software-properties-common && \
    pip install --upgrade pip

# Install Basic Packages