### alpine-iptables

Serves as the base image for `k8s.gcr.io/kube-proxy-${ARCH}`

This image is compiled for multiple architectures.

#### How to release

If you're editing the Dockerfile or some other thing, please bump the `TAG` in the Makefile.

```console
Build and  push images for all the architectures
$ make all-push
# ---> staging-k8s.gcr.io/alpine-iptables-amd64:TAG
# ---> staging-k8s.gcr.io/alpine-iptables-arm:TAG
# ---> staging-k8s.gcr.io/alpine-iptables-arm64:TAG
# ---> staging-k8s.gcr.io/alpine-iptables-ppc64le:TAG
# ---> staging-k8s.gcr.io/alpine-iptables-s390x:TAG
```

If you don't want to push the images, run `make build ARCH={target_arch}` or `make all-build` instead


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/build/alpine-iptables/README.md?pixel)]()
