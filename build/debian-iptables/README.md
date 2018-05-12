### debian-iptables

Serves as the base image for `k8s.gcr.io/kube-proxy-${ARCH}` and multiarch (not `amd64`) `k8s.gcr.io/flannel-${ARCH}` images.

This image is compiled for multiple architectures.

#### How to release

If you're editing the Dockerfile or some other thing, please bump the `TAG` in the Makefile.

```console
# Build for linux/amd64 (default)
$ make push ARCH=amd64
# ---> staging-k8s.gcr.io/debian-iptables-amd64:TAG

$ make push ARCH=arm
# ---> staging-k8s.gcr.io/debian-iptables-arm:TAG

$ make push ARCH=arm64
# ---> staging-k8s.gcr.io/debian-iptables-arm64:TAG

$ make push ARCH=ppc64le
# ---> staging-k8s.gcr.io/debian-iptables-ppc64le:TAG

$ make push ARCH=s390x
# ---> staging-k8s.gcr.io/debian-iptables-s390x:TAG
```

If you don't want to push the images, run `make` or `make build` instead


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/build/debian-iptables/README.md?pixel)]()
