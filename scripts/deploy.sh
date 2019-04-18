#!/bin/sh

if [ "$(docker ps -q -f name=nlp_backend)" ]; then
  docker stop nlp_backend
fi

if [ "$(docker ps -aq -f name=nlp_backend)" ]; then
  docker rm nlp_backend
fi

docker build -f Dockerfile.linux -t nlp_backend .
docker run -d -p 6543:6543 --name nlp_backend nlp_backend
