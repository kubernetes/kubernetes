### etcd

This is a small etcd image used in Kubernetes setups where `etcd` is deployed as a docker image.

For `amd64`, official `etcd` and `etcdctl` binaries are downloaded from Github to maintain official support.
For other architectures, `etcd` is cross-compiled from source. Arch-specific `busybox` images serve as base images.

#### How to release

```console
# Build for linux/amd64 (default)
$ make push ARCH=amd64
# ---> gcr.kubernetes.io/etcd-amd64:TAG
# ---> gcr.kubernetes.io/etcd:TAG

$ make push ARCH=arm
# ---> gcr.kubernetes.io/etcd-arm:TAG

$ make push ARCH=arm64
# ---> gcr.kubernetes.io/etcd-arm64:TAG

$ make push ARCH=ppc64le
# ---> gcr.kubernetes.io/etcd-ppc64le:TAG
```

If you don't want to push the images, run `make` or `make build` instead


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/images/etcd/README.md?pixel)]()
