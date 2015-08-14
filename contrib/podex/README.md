# podex

## Description

`podex` is a command line tool to bootstrap a kubernetes container manifests from docker image metadata.

Manifests can then be edited by a human to match deployment needs.

## Usage
```
$ podex [-daemon] [-insecure-registry] [-insecure-skip-verify] [-format yaml|json] [-type=pod|container] [-name PODNAME] IMAGES...

```

### Options
- `format`: manifest format to output, `yaml` (default) or `json`
- `type`: manifest type to output, `pod` (default) or `container`
- `name`: manifest name (required with multiple images, optional with single image: default to image base name)

### Flags
- `daemon`: run in daemon mode
- `insecure-registry`: connect to insecure registry using HTTP
- `insecure-skip-verify`: skip registry certificate verify (registry with self-signed certificate)

### Examples
```
$ podex google/nodejs-hello > pod.yaml
$ podex -format json -type container -name nodejs-nginx google/nodejs-hello nginx > container.json
```

## TODOs
- [ ] option to merge multiple container manifest into one pod
- [ ] docker run flags support
- [ ] option to generate service bindings from links


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/podex/README.md?pixel)]()
