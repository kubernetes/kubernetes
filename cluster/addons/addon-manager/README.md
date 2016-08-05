### addon-manager

The `addon-manager` periodically checks for Kubernetes manifest changes in the `/etc/kubernetes/addons` directory,
and when there's a new or changed addon, the `addon-manager` automatically `kubectl create`s it.

It supports `ReplicationControllers`, `Deployments`, `DaemonSets`, `ConfigMaps`, `Services`, `PersistentVolumes` and
`PersistentVolumeClaims`.

The `addon-manager` is built for multiple architectures.

#### How to release

1. Change something in the source
2. Bump `VERSION` in the `Makefile`
3. Bump `KUBECTL_VERSION` in the `Makefile` if required
4. Build the `amd64` image and test it on a cluster
5. Push all images

```console
# Build for linux/amd64 (default)
$ make push ARCH=amd64
# ---> gcr.io/google-containers/kube-addon-manager-amd64:VERSION
# ---> gcr.io/google-containers/kube-addon-manager:VERSION (image with backwards-compatible naming)

$ make push ARCH=arm
# ---> gcr.io/google-containers/kube-addon-manager-arm:VERSION

$ make push ARCH=arm64
# ---> gcr.io/google-containers/kube-addon-manager-arm64:VERSION

$ make push ARCH=ppc64le
# ---> gcr.io/google-containers/kube-addon-manager-ppc64le:VERSION
```

If you don't want to push the images, run `make` or `make build` instead


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/addon-manager/README.md?pixel)]()
