#!/bin/sh

if [ "$(docker ps -q -f name=nlp_backend)" ]; then
  docker stop nlp_backend
fi

if [ "$(docker ps -aq -f name=nlp_backend)" ]; then
  docker rm nlp_backend
fi
