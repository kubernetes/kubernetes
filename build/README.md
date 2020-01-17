# Building Kubernetes

Building Kubernetes is easy if you take advantage of the containerized build environment. This document will help guide you through understanding this build process.

## Requirements

1. Docker, using one of the following configurations:
  * **macOS** You can either use Docker for Mac or docker-machine. See installation instructions [here](https://docs.docker.com/docker-for-mac/).
     **Note**: You will want to set the Docker VM to have at least 4.5GB of initial memory or building will likely fail. (See: [#11852]( http://issue.k8s.io/11852)).
  * **Linux with local Docker**  Install Docker according to the [instructions](https://docs.docker.com/installation/#installation) for your OS.
  * **Remote Docker engine** Use a big machine in the cloud to build faster. This is a little trickier so look at the section later on.
2. **Optional** [Google Cloud SDK](https://developers.google.com/cloud/sdk/)

You must install and configure Google Cloud SDK if you want to upload your release to Google Cloud Storage and may safely omit this otherwise.

## Overview

While it is possible to build Kubernetes using a local golang installation, we have a build process that runs in a Docker container.  This simplifies initial set up and provides for a very consistent build and test environment.

## Key scripts

The following scripts are found in the [`build/`](.) directory. Note that all scripts must be run from the Kubernetes root directory.

* [`build/run.sh`](run.sh): Run a command in a build docker container.  Common invocations:
  *  `build/run.sh make`: Build just linux binaries in the container.  Pass options and packages as necessary.
  *  `build/run.sh make cross`: Build all binaries for all platforms
  *  `build/run.sh make kubectl KUBE_BUILD_PLATFORMS=darwin/amd64`: Build the specific binary for the specific platform (`kubectl` and `darwin/amd64` respectively in this example)
  *  `build/run.sh make test`: Run all unit tests
  *  `build/run.sh make test-integration`: Run integration test
  *  `build/run.sh make test-cmd`: Run CLI tests
* [`build/copy-output.sh`](copy-output.sh): This will copy the contents of `_output/dockerized/bin` from the Docker container to the local `_output/dockerized/bin`. It will also copy out specific file patterns that are generated as part of the build process. This is run automatically as part of `build/run.sh`.
* [`build/make-clean.sh`](make-clean.sh): Clean out the contents of `_output`, remove any locally built container images and remove the data container.
* [`build/shell.sh`](shell.sh): Drop into a `bash` shell in a build container with a snapshot of the current repo code.

## Basic Flow

The scripts directly under [`build/`](.) are used to build and test.  They will ensure that the `kube-build` Docker image is built (based on [`build/build-image/Dockerfile`](build-image/Dockerfile)) and then execute the appropriate command in that container.  These scripts will both ensure that the right data is cached from run to run for incremental builds and will copy the results back out of the container.

The `kube-build` container image is built by first creating a "context" directory in `_output/images/build-image`.  It is done there instead of at the root of the Kubernetes repo to minimize the amount of data we need to package up when building the image.

There are 3 different containers instances that are run from this image.  The first is a "data" container to store all data that needs to persist across to support incremental builds. Next there is an "rsync" container that is used to transfer data in and out to the data container.  Lastly there is a "build" container that is used for actually doing build actions.  The data container persists across runs while the rsync and build containers are deleted after each use.

`rsync` is used transparently behind the scenes to efficiently move data in and out of the container.  This will use an ephemeral port picked by Docker.  You can modify this by setting the `KUBE_RSYNC_PORT` env variable.

All Docker names are suffixed with a hash derived from the file path (to allow concurrent usage on things like CI machines) and a version number.  When the version number changes all state is cleared and clean build is started.  This allows the build infrastructure to be changed and signal to CI systems that old artifacts need to be deleted.

## Proxy Settings

If you are behind a proxy and you are letting these scripts use `docker-machine` to set up your local VM for you on macOS, you need to export proxy settings for Kubernetes build, the following environment variables should be defined.

```
export KUBERNETES_HTTP_PROXY=http://username:password@proxyaddr:proxyport
export KUBERNETES_HTTPS_PROXY=https://username:password@proxyaddr:proxyport
```

Optionally, you can specify addresses of no proxy for Kubernetes build, for example

```
export KUBERNETES_NO_PROXY=127.0.0.1
```

If you are using sudo to make Kubernetes build for example make quick-release, you need run `sudo -E make quick-release` to pass the environment variables.

## Really Remote Docker Engine

It is possible to use a Docker Engine that is running remotely (under your desk or in the cloud).  Docker must be configured to connect to that machine and the local rsync port must be forwarded (via SSH or nc) from localhost to the remote machine.

To do this easily with GCE and `docker-machine`, do something like this:
```
# Create the remote docker machine on GCE.  This is a pretty beefy machine with SSD disk.
KUBE_BUILD_VM=k8s-build
KUBE_BUILD_GCE_PROJECT=<project>
docker-machine create \
  --driver=google \
  --google-project=${KUBE_BUILD_GCE_PROJECT} \
  --google-zone=us-west1-a \
  --google-machine-type=n1-standard-8 \
  --google-disk-size=50 \
  --google-disk-type=pd-ssd \
  ${KUBE_BUILD_VM}

# Set up local docker to talk to that machine
eval $(docker-machine env ${KUBE_BUILD_VM})

# Pin down the port that rsync will be exposed on the remote machine
export KUBE_RSYNC_PORT=8730

# forward local 8730 to that machine so that rsync works
docker-machine ssh ${KUBE_BUILD_VM} -L ${KUBE_RSYNC_PORT}:localhost:${KUBE_RSYNC_PORT} -N &
```

Look at `docker-machine stop`, `docker-machine start` and `docker-machine rm` to manage this VM.

## Releasing

The [`build/release.sh`](release.sh) script will build a release.  It will build binaries, run tests, (optionally) build runtime Docker images.

The main output is a tar file: `kubernetes.tar.gz`.  This includes:
* Cross compiled client utilities.
* Script (`kubectl`) for picking and running the right client binary based on platform.
* Examples
* Cluster deployment scripts for various clouds
* Tar file containing all server binaries

In addition, there are some other tar files that are created:
* `kubernetes-client-*.tar.gz` Client binaries for a specific platform.
* `kubernetes-server-*.tar.gz` Server binaries for a specific platform.

When building final release tars, they are first staged into `_output/release-stage` before being tar'd up and put into `_output/release-tars`.

## Reproducibility
`make release`, its variant `make quick-release`, and Bazel all provide a
hermetic build environment which should provide some level of reproducibility
for builds. `make` itself is **not** hermetic.

The Kubernetes build environment supports the [`SOURCE_DATE_EPOCH` environment
variable](https://reproducible-builds.org/specs/source-date-epoch/) specified by
the Reproducible Builds project, which can be set to a UNIX epoch timestamp.
This will be used for the build timestamps embedded in compiled Go binaries,
and maybe someday also Docker images.

One reasonable setting for this variable is to use the commit timestamp from the
tip of the tree being built; this is what the Kubernetes CI system uses. For
example, you could use the following one-liner:

```bash
SOURCE_DATE_EPOCH=$(git show -s --format=format:%ct HEAD)
```

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/build/README.md?pixel)]()
