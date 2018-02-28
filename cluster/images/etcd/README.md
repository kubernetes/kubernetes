### etcd

This is a small etcd image used in Kubernetes setups where `etcd` is deployed as a docker image.

For `amd64`, official `etcd` and `etcdctl` binaries are downloaded from Github to maintain official support.
For other architectures, `etcd` is cross-compiled from source. Arch-specific `busybox` images serve as base images.

#### How to release

First, run the migration and rollback tests.

```console
$ make build test
```

Next, build and push the docker images for all supported architectures.

```console
# Build for linux/amd64 (default)
$ make push ARCH=amd64
# ---> staging-k8s.gcr.io/etcd-amd64:TAG
# ---> staging-k8s.gcr.io/etcd:TAG

$ make push ARCH=arm
# ---> staging-k8s.gcr.io/etcd-arm:TAG

$ make push ARCH=arm64
# ---> staging-k8s.gcr.io/etcd-arm64:TAG

$ make push ARCH=ppc64le
# ---> staging-k8s.gcr.io/etcd-ppc64le:TAG

$ make push ARCH=s390x
# ---> staging-k8s.gcr.io/etcd-s390x:TAG
```

If you don't want to push the images, run `make` or `make build` instead


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/images/etcd/README.md?pixel)]()
