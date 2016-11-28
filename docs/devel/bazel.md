# Build with Bazel

Building with bazel is currently experimental. Automanaged BUILD rules have the
tag "automanaged" and are maintained by
[gazel](https://github.com/mikedanese/gazel). Instructions for installing bazel
can be found [here](https://www.bazel.io/versions/master/docs/install.html).

To build docker images for the components, run:

```
$ bazel build //build-tools/...
```

To run many of the unit tests, run:

```
$ bazel test //cmd/... //build-tools/... //pkg/... //federation/... //plugin/...
```

To update automanaged build files, run:

```
$ ./hack/update-bazel.sh
```

**NOTES**: `update-bazel.sh` only works if check out directory of Kubernetes is "$GOPATH/src/k8s.io/kubernetes".

To update a single build file, run:

```
$ # get gazel
$ go get -u github.com/mikedanese/gazel
$ # .e.g. ./pkg/kubectl/BUILD
$ gazel -root="${YOUR_KUBE_ROOT_PATH}" ./pkg/kubectl
```

Updating BUILD file for a package will be required when:
* Files are added to or removed from a package
* Import dependencies change for a package


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/bazel.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
