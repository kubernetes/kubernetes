# Cluster Management

This doc is in progress.

## Upgrading a cluster

The `cluster/kube-push.sh` script will do a rudimentary update; it is a 1.0 roadmap item to have a robust live cluster update system.

## Updgrading to a different API version

There is a sequence of steps to upgrade to a new API version.

1. Turn on the new version.
2. Upgrade the cluster's storage to use the new version.
3. Upgrade all config files. Identify users of the old api version endpoints.
3. Turn off the old version.

### Turn on or off an API version for your cluster

TODO: There's an apiserver flag for this.

### Switching your cluster's storage API version

TODO: This functionality hasn't been written yet.

### Switching your config files to a new API version

You can use the kube-version-change utility to convert config files between different API versions.

```
$ hack/build-go.sh cmd/kube-version-change
$ _output/go/bin/kube-version-change -i myPod.v1beta1.yaml -o myPod.v1beta3.yaml
```
