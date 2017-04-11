#!/bin/bash

rm -rf ./documents/*
cp /manifest/manifest.json ./manifest.json
cp /source/* ./documents/
node brodoc.js
cp -r ./* /build/
