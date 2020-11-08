#!/bin/bash

set -eu -o pipefail

BRANCH="${TRAVIS_BRANCH//\//-}"

if [ "$BRANCH" = "master" ] && [ "$TRAVIS_PULL_REQUEST" = "false" ]; then
    echo "1.0.$TRAVIS_BUILD_NUMBER"
else
    echo "0.1.$TRAVIS_BUILD_NUMBER-$BRANCH"
fi