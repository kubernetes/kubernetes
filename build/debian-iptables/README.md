### debian-iptables

Serves as the base image for `k8s.gcr.io/kube-proxy-${ARCH}` and multiarch (not `amd64`) `k8s.gcr.io/flannel-${ARCH}` images.

This image is compiled for multiple architectures.

#### How to release

If you're editing the Dockerfile or some other thing, please bump the `TAG` in the Makefile.

```console
Build and  push images for all the architectures
$ make all-push
# ---> staging-k8s.gcr.io/debian-iptables-amd64:TAG
# ---> staging-k8s.gcr.io/debian-iptables-arm:TAG
# ---> staging-k8s.gcr.io/debian-iptables-arm64:TAG
# ---> staging-k8s.gcr.io/debian-iptables-ppc64le:TAG
# ---> staging-k8s.gcr.io/debian-iptables-s390x:TAG
```

If you don't want to push the images, run `make build ARCH={target_arch}` or `make all-build` instead


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/build/debian-iptables/README.md?pixel)]()
