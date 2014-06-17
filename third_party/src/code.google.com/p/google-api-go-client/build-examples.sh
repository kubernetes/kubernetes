#!/bin/sh

set -e

make -C google-api install
make -C google-api-go-generator install
google-api-go-generator/google-api-go-gen -cache -api=tasks:v1
google-api-go-generator/google-api-go-gen -cache -api=urlshortener:v1
make -C tasks/v1 install
make -C urlshortener/v1 install
make -C examples
