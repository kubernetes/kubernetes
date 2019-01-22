## serve_hostname

This is a small util app to serve your hostname on TCP and/or UDP.  Useful for testing.

The `serve_hostname` Makefile supports multiple architectures, which means it may cross-compile and build an docker image easily.
Arch-specific busybox images serve as base images.

If you are releasing a new version, please bump the `TAG` value in the `Makefile` before building the images.

## How to release:

```
# Build cross-platform binaries
$ make all-push

# Build for linux/amd64 (default)
$ make push ARCH=amd64
# ---> staging-k8s.gcr.io/serve_hostname-amd64:TAG

$ make push ARCH=arm
# ---> staging-k8s.gcr.io/serve_hostname-arm:TAG

$ make push ARCH=arm64
# ---> staging-k8s.gcr.io/serve_hostname-arm64:TAG

$ make push ARCH=ppc64le
# ---> staging-k8s.gcr.io/serve_hostname-ppc64le:TAG

$ make push ARCH=s390x
# ---> staging-k8s.gcr.io/serve_hostname-s390x:TAG
```

Of course, if you don't want to push the images, run `make all-container` or `make container ARCH={target_arch}` instead.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/for-demos/serve_hostname/README.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/test/images/serve_hostname/README.md?pixel)]()
