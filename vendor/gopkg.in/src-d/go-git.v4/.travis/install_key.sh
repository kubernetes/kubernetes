#!/bin/bash
openssl aes-256-cbc \
    -K $encrypted_1477e58fe67a_key \
    -iv $encrypted_1477e58fe67a_iv \
    -in .travis/deploy.pem.enc \
    -out $HOME/.travis/deploy.pem -d

chmod 600 $HOME/.travis/deploy.pem

