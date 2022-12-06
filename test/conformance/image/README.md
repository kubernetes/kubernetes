### conformance

`conformance` is a standalone container to launch Kubernetes end-to-end tests, for the purposes of conformance testing.
`conformance` is built for multiple architectures and _the image is pushed automatically on every release._

#### How to release by hand

```console
# First, build the binaries by running make from the root directory
$ make WHAT="test/e2e/e2e.test github.com/onsi/ginkgo/v2/ginkgo cmd/kubectl test/conformance/image/go-runner"

# Build for linux/amd64 (default)
# export REGISTRY=$HOST/$ORG to switch from registry.k8s.io

$ make push VERSION={target_version} ARCH=amd64
# ---> registry.k8s.io/conformance-amd64:VERSION
# ---> registry.k8s.io/conformance:VERSION (image with backwards-compatible naming)

$ make push VERSION={target_version} ARCH=arm
# ---> registry.k8s.io/conformance-arm:VERSION

$ make push VERSION={target_version} ARCH=arm64
# ---> registry.k8s.io/conformance-arm64:VERSION

$ make push VERSION={target_version} ARCH=ppc64le
# ---> registry.k8s.io/conformance-ppc64le:VERSION

$ make push VERSION={target_version} ARCH=s390x
# ---> registry.k8s.io/conformance-s390x:VERSION
```

If you don't want to push the images, run `make` or `make build` instead


#### How to run tests

```
kubectl create -f conformance-e2e.yaml
```

