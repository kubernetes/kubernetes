# Replication controller tools

## resize.sh
Resizes a replication controller to the specified number of pods.
```
$ resize.sh
usage: resize.sh <replication controller name> <size>
$ resize.sh redisslave 4
```

## stop.sh
Resizes a replication controller to 0 pods and waits until the pods are deleted.
```
$ stop.sh
usage: stop.sh <replication controller name>
$ stop.sh redisslave
```
