#!/bin/bash
# Build a Docker image for running a version of fluentd
# configure with an Elasticsearch plug-in which processes
# Docker log files.

set -e 

docker build -t satnam6502/docker-fluentd .