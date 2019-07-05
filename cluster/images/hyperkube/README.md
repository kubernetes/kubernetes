### hyperkube

`hyperkube` is an all-in-one binary for the Kubernetes server components.

`hyperkube` is built for multiple architectures and _the image is pushed automatically on every release._

#### How to release by hand

```console
# First, build the binaries
$ build/run.sh make cross

# Build for linux/amd64 (default)
# export REGISTRY=$HOST/$ORG to switch from staging-k8s.gcr.io

$ make push VERSION={target_version} ARCH=amd64
# ---> staging-k8s.gcr.io/hyperkube-amd64:VERSION
# ---> staging-k8s.gcr.io/hyperkube:VERSION (image with backwards-compatible naming)

$ make push VERSION={target_version} ARCH=arm
# ---> staging-k8s.gcr.io/hyperkube-arm:VERSION

$ make push VERSION={target_version} ARCH=arm64
# ---> staging-k8s.gcr.io/hyperkube-arm64:VERSION

$ make push VERSION={target_version} ARCH=ppc64le
# ---> staging-k8s.gcr.io/hyperkube-ppc64le:VERSION

$ make push VERSION={target_version} ARCH=s390x
# ---> staging-k8s.gcr.io/hyperkube-s390x:VERSION
```

If you don't want to push the images, run `make` or `make build` instead


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/images/hyperkube/README.md?pixel)]()
