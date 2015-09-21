# podex

## Description

`podex` is a command line tool to bootstrap a kubernetes container manifests from docker image metadata.

Manifests can then be edited by a human to match deployment needs.

## Usage
```
$ podex [-json|-yaml] [-id PODNAME] IMAGES...

$ podex -json [-id PODNAME] IMAGES... > pod.json
$ podex -yaml [-id PODNAME] IMAGES... > pod.yaml
```

### Options
- `id`: set the pod name (required with multiple images, optional with single image: default to image base name)
- `json`: puts the container manifest into JSON format
- `yaml`: puts the container manifest into YAML format

### Examples
```
$ podex -json google/nodejs-hello > pod.yaml
$ podex -yaml -id nodejs-nginx google/nodejs-hello nginx > pod.yaml
```

## TODOs
- [ ] option generate a full pod manifest (w/ `desired state`)
- [ ] option to merge multiple container manifest into one pod
- [ ] docker run flags support
- [ ] option to generate service bindings from links
