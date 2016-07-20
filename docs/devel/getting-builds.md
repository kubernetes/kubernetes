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

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/devel/getting-builds.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Getting Kubernetes Builds

You can use [hack/get-build.sh](http://releases.k8s.io/HEAD/hack/get-build.sh)
to get a build or to use as a reference on how to get the most recent builds
with curl. With `get-build.sh` you can grab the most recent stable build, the
most recent release candidate, or the most recent build to pass our ci and gce
e2e tests (essentially a nightly build).

Run `./hack/get-build.sh -h` for its usage.

To get a build at a specific version (v1.1.1) use:

```console
./hack/get-build.sh v1.1.1
```

To get the latest stable release:

```console
./hack/get-build.sh release/stable
```

Use the "-v" option to print the version number of a build without retrieving
it. For example, the following prints the version number for the latest ci
build:

```console
./hack/get-build.sh -v ci/latest
```

You can also use the gsutil tool to explore the Google Cloud Storage release
buckets. Here are some examples:

```sh
gsutil cat gs://kubernetes-release-dev/ci/latest.txt          # output the latest ci version number
gsutil cat gs://kubernetes-release-dev/ci/latest-green.txt    # output the latest ci version number that passed gce e2e
gsutil ls gs://kubernetes-release-dev/ci/v0.20.0-29-g29a55cc/ # list the contents of a ci release
gsutil ls gs://kubernetes-release/release                 # list all official releases and rcs
```

## Install `gsutil`

Example installation:

```console
$ curl -sSL https://storage.googleapis.com/pub/gsutil.tar.gz | sudo tar -xz -C /usr/local/src
$ sudo ln -s /usr/local/src/gsutil/gsutil /usr/bin/gsutil
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/getting-builds.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
