### flannel

This is used mostly for the `docker-multinode` config, but also in other places where flannel runs in a container.

For `amd64`, this image equals to `quay.io/coreos/flannel` to maintain official support.
For other architectures, `flannel` is cross-compiled. The `debian-iptables` image serves as base image.

#### How to release

```console
# Build for linux/amd64 (default)
$ make push ARCH=amd64
# ---> gcr.io/google_containers/flannel-amd64:TAG

$ make push ARCH=arm
# ---> gcr.io/google_containers/flannel-arm:TAG

$ make push ARCH=arm64
# ---> gcr.io/google_containers/flannel-arm64:TAG

$ make push ARCH=ppc64le
# ---> gcr.io/google_containers/flannel-ppc64le:TAG
```

If you don't want to push the images, run `make` or `make build` instead


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/images/flannel/README.md?pixel)]()
