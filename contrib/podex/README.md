# podex

## Description

`podex` is a command line tool to bootstrap a kubernetes container manifests from docker image metadata.

Manifests can then be edited by a human to match deployment needs.

## Usage
```
$ podex [-format json|yaml] [-type=pod|container] [-name PODNAME] IMAGES...

```

### Options
- `format`: manifest format to output, `yaml` (default) or `json`
- `json`: manifest type to output, `pod` (default) or `container`
- `name`: manifest name (required with multiple images, optional with single image: default to image base name)

### Examples
```
$ podex google/nodejs-hello > pod.yaml
$ podex -format json -type container -name nodejs-nginx google/nodejs-hello nginx > container.json
```

## TODOs
- [ ] option to merge multiple container manifest into one pod
- [ ] docker run flags support
- [ ] option to generate service bindings from links
