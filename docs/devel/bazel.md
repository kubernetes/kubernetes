<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

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
