<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Getting Kubernetes Builds

You can use [hack/get-build.sh](http://releases.k8s.io/release-1.1/hack/get-build.sh) to or use as a reference on how to get the most recent builds with curl. With `get-build.sh` you can grab the most recent stable build, the most recent release candidate, or the most recent build to pass our ci and gce e2e tests (essentially a nightly build).

```console
usage:
  ./hack/get-build.sh [stable|release|latest|latest-green]

        stable:        latest stable version
        release:       latest release candidate
        latest:        latest ci build
        latest-green:  latest ci build to pass gce e2e
```

You can also use the gsutil tool to explore the Google Cloud Storage release bucket. Here are some examples:

```sh
gsutil cat gs://kubernetes-release/ci/latest.txt          # output the latest ci version number
gsutil cat gs://kubernetes-release/ci/latest-green.txt    # output the latest ci version number that passed gce e2e
gsutil ls gs://kubernetes-release/ci/v0.20.0-29-g29a55cc/ # list the contents of a ci release
gsutil ls gs://kubernetes-release/release                 # list all official releases and rcs
```




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/getting-builds.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
