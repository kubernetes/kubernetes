#!/bin/bash

set -e
set -x

# Statically build cAdvisor from source and stage it.
go build --ldflags '-extldflags "-static"' github.com/google/cadvisor
