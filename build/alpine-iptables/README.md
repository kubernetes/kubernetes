### alpine-iptables

Serves as the base image for `gcr.io/google_containers/kube-proxy-${ARCH}` and multiarch (not `amd64`) images.

This image is compiled for multiple architectures.

#### How to release

If you're editing the Dockerfile or some other thing, please bump the `TAG` in the Makefile.

```console
# Build for linux/amd64 (default)
$ make push ARCH=amd64
# ---> gcr.io/google_containers/alpine-iptables-amd64:TAG

$ make push ARCH=arm
# ---> gcr.io/google_containers/alpine-iptables-arm:TAG

$ make push ARCH=arm64
# ---> gcr.io/google_containers/alpine-iptables-arm64:TAG

# Currently, ppc64le and s390x alpine images do not exist.
$ make push ARCH=ppc64le
# ---> gcr.io/google_containers/alpine-iptables-ppc64le:TAG

$ make push ARCH=s390x
# ---> gcr.io/google_containers/alpine-iptables-s390x:TAG
```

If you don't want to push the images, run `make` or `make build` instead


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/build/alpine-iptables/README.md?pixel)]()
