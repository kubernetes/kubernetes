# podex

## Description

`podex` is a command line tool to bootstrap a kubernetes container manifests from docker image metadata.

Manifests can then be edited by a human to match deployment needs.

## Usage
```
$ docker pull google/nodejs-hello
$ podex -yaml google/nodejs-hello > pod.yaml
$ podex -json google/nodejs-hello > pod.json
```
