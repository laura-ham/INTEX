#!/usr/bin/env bash
set -ex

pip3 install awscli
eval $(aws ecr get-login --region eu-west-1 --no-include-email)
docker pull ${ECR_REPOSITORY}:latest || true