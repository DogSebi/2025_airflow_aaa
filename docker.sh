#!/bin/bash

IMAGE_NAME=dogsebi/airflow_2025

docker build -t $IMAGE_NAME .
docker push $IMAGE_NAME