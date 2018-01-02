#!/bin/bash
# Run the build utility via Docker

set -e

# Make sure our working dir is the dir of the script
DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
cd $DIR


# Build new docker image
docker build -f Dockerfile_build_ubuntu64 -t influxdb-builder $DIR
echo "Running build.py"
# Run docker
docker run --rm \
    -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    -v $HOME/.aws.conf:/root/.aws.conf \
    -v $DIR:/root/go/src/github.com/influxdata/influxdb \
    influxdb-builder \
    "$@"

