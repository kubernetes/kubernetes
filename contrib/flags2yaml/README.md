# flags2yaml

`flags2yaml` is a command-line tool to generate flat YAML from command-line flags

### Usage
```
$ flags2yaml image=dockerfile/nginx | simplegen - | cluster/kubectl.sh create -f -
```
