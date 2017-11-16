### debian-iptables

Serves as the base image for `gcr.io/google_containers/kube-proxy-${ARCH}` and multiarch (not `amd64`) `gcr.io/google_containers/flannel-${ARCH}` images.

This image is compiled for multiple architectures.

#### How to release

If you're editing the Dockerfile or some other thing, please bump the `TAG` in the Makefile.

```console
# Build for linux/amd64 (default)
$ make push ARCH=amd64
# ---> gcr.io/google_containers/debian-iptables-amd64:TAG

$ make push ARCH=arm
# ---> gcr.io/google_containers/debian-iptables-arm:TAG

$ make push ARCH=arm64
# ---> gcr.io/google_containers/debian-iptables-arm64:TAG

$ make push ARCH=ppc64le
# ---> gcr.io/google_containers/debian-iptables-ppc64le:TAG

$ make push ARCH=s390x
# ---> gcr.io/google_containers/debian-iptables-s390x:TAG
```

If you don't want to push the images, run `make` or `make build` instead


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/build/debian-iptables/README.md?pixel)]()
