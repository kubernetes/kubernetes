# Build with Bazel

Building with bazel is currently experimental. Automanaged BUILD rules have the
tag "automanaged" and are maintained by
[gazel](https://github.com/mikedanese/gazel). Instructions for installing bazel
can be found [here](https://www.bazel.io/versions/master/docs/install.html).

To build docker images for the components, run:

```
$ bazel build //build/...
```

To run many of the unit tests, run:

```
$ bazel test //cmd/... //build/... //pkg/... //federation/... //plugin/...
```

To update automanaged build files, run:

```
$ ./hack/update-bazel.sh
```


To update a single build file, run:

```
$ # get gazel
$ go get -u github.com/mikedanese/gazel
$ # .e.g. ./pkg/kubectl/BUILD
$ gazel ./pkg/kubectl
```

Updating BUILD file for a package will be required when:
* Files are added to or removed from a package
* Import dependencies change for a package


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/bazel.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
