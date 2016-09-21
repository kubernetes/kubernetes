### flannel

This image is used for building stable flannel releases for use with Kubernetes as well as producing experimental images for testing before flannel releases a new version.

#### How to release

```console
# Build and push the flannel image for linux/amd64, linux/arm, linux/arm64 and linux/ppc64le 
$ make push-all
# ---> gcr.io/google_containers/flannel-amd64:TAG
# ---> gcr.io/google_containers/flannel-arm:TAG
# ---> gcr.io/google_containers/flannel-arm64:TAG
# ---> gcr.io/google_containers/flannel-ppc64le:TAG
```

#### Changelog


 - 0.6.2-kube: The official 0.6.1 release plus the https://github.com/coreos/flannel/pull/483 PR which makes it possible to use flannel as a DaemonSet on every node. For an example DaemonSet manifest, see flannel-ds.yaml in this directory.
 - 0.5.5: Stable release of flannel for all architectures. On amd64, the official `quay.io/coreos/flannel` image is used. For the other architectures, flannel is cross-compiled

If you don't want to push the images, run `make build-all` instead

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/images/flannel/README.md?pixel)]()
