### hyperkube

`hyperkube` is an all-in-one binary for the Kubernetes server components
Also, it's very easy to run this `hyperkube` setup dockerized.
See http://kubernetes.io/docs/getting-started-guides/docker/ for up-to-date commands.

`hyperkube` is built for multiple architectures and pushed on every release.

#### How to release by hand

```console
# First, build the 
$ build/run.sh hack/build-cross.sh

# Build for linux/amd64 (default)
$ make push VERSION={target_version} ARCH=amd64
# ---> gcr.io/google_containers/hyperkube-amd64:VERSION
# ---> gcr.io/google_containers/hyperkube:VERSION (image with backwards-compatible naming)

$ make push VERSION={target_version} ARCH=arm
# ---> gcr.io/google_containers/hyperkube-arm:VERSION
```

If you don't want to push the images, run `make` or `make build` instead


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/images/hyperkube/README.md?pixel)]()
