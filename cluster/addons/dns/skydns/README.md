# skydns for kubernetes
=======================

This container only exists until skydns itself is reduced in some way. At the
time of this writing, it is over 600 MB large.

#### How to release

This image is compiled for multiple architectures.
If you're rebuilding the image, please bump the `TAG` in the Makefile.

```console
# Build for linux/amd64 (default)
$ make push ARCH=amd64
# ---> gcr.io/google_containers/skydns-amd64:TAG
# ---> gcr.io/google_containers/skydns:TAG (image with backwards-compatible naming)

$ make push ARCH=arm
# ---> gcr.io/google_containers/skydns-arm:TAG

$ make push ARCH=arm64
# ---> gcr.io/google_containers/skydns-arm64:TAG

$ make push ARCH=ppc64le
# ---> gcr.io/google_containers/skydns-ppc64le:TAG
```

If you don't want to push the images, run `make` or `make build` instead

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/dns/skydns/README.md?pixel)]()
