### debian-hyperkube-base

Serves as the base image for `k8s.gcr.io/hyperkube-${ARCH}`
images.

This image is compiled for multiple architectures.

#### How to release

If you're editing the Dockerfile or some other thing, please bump the `TAG` in the Makefile.

```console
# Build for linux/amd64 (default)
$ make push ARCH=amd64
# ---> k8s.gcr.io/debian-hyperkube-base-amd64:TAG

$ make push ARCH=arm
# ---> k8s.gcr.io/debian-hyperkube-base-arm:TAG

$ make push ARCH=arm64
# ---> k8s.gcr.io/debian-hyperkube-base-arm64:TAG

$ make push ARCH=ppc64le
# ---> k8s.gcr.io/debian-hyperkube-base-ppc64le:TAG

$ make push ARCH=s390x
# ---> k8s.gcr.io/debian-hyperkube-base-s390x:TAG
```

If you don't want to push the images, run `make build` instead


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/build/debian-hyperkube-base/README.md?pixel)]()
