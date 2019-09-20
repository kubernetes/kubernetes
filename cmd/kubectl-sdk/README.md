## Overview

This `kubectl` target is the binary that is packaged and distributed through gcloud.

### Why are there two versions of the kubectl target ?

The standard `kubectl` target (in the `cmd/kubectl` directory) is dynamically linked
with the `BoringCrypto` library in order to be FIPS compliant. The `kubectl` target
in this directory is built statically (required for gcloud distribution), and it
includes the dispatcher functionality. This target is referred to as the
`kubectl-dispatcher`, and it is designed to solve the **version skew problem**.

### What is the "version skew problem" ?

The **version skew problem** arises when the version of `kubectl` significantly
differs from the version of the API Server in the cluster that it is
communicating with. Generally, only a `kubectl` that is +/- one minor version
distant from the API Server is guaranteed to work. For example, with a 1.17 API
Server, `kubectl` versions 1.16, 1.17, and 1.18 are supported. The **version
skew problem** can significantly impede the velocity of GKE releases, since GKE
releases must account for the single `kubectl` version shipped. This
`kubectl-dispatcher` target addresses this problem by shipping multiple versions
of `kubectl`, matching the `kubectl` version to the version of the API
Server.

## Building

This target is built using the `build-kubectl-dispatcher` script in this
directory. The parameter to this script is the name of the GKE Kubernetes
release branch to build the `kubectl-dispatcher` at. Example:

```bash
$ ./build-kubectl-dispatcher release-1.17.17-gke.55
```

This release branch is determined from the current major/minor version of the
default GKE cluster:

```bash
$ gcloud container get-server-config --zone=us-central1
Fetching server config for us-central1
...
defaultClusterVersion: 1.17.17-gke.1101
```

In this case, the default cluster is version 1.17. We can determine the 1.17
release branch from the **Branches** left nav of the GKE Kubernetes source code
(click **More...**). In this case we can see the latest patch version of the
1.17 release branch is
[release-1.17.17-gke.55](https://gke-internal.git.corp.google.com/kubernetes/+/refs/heads/release-1.17.17-gke.55).

This script will clone this release branch and build the dispatcher binaries using
the `cross-compile` script (also in this directory). The cross-compiled
`kubectl-dispatcher` artifacts are then stored in the `gs://kubectl-dispatcher`
bucket. Example:

```bash
$ gsutil ls -l gs://kubectl-dispatcher/v1.17.17/release
  13286943  2021-03-29T21:00:37Z  gs://kubectl-dispatcher/v1.17.17/release/kubectl-dispatcher-darwin-386.tar.gz
  13820496  2021-03-29T20:59:34Z  gs://kubectl-dispatcher/v1.17.17/release/kubectl-dispatcher-darwin-amd64.tar.gz
  12607583  2021-03-29T20:58:32Z  gs://kubectl-dispatcher/v1.17.17/release/kubectl-dispatcher-linux-386.tar.gz
  13127171  2021-03-29T20:57:30Z  gs://kubectl-dispatcher/v1.17.17/release/kubectl-dispatcher-linux-amd64.tar.gz
  12652011  2021-03-29T21:02:42Z  gs://kubectl-dispatcher/v1.17.17/release/kubectl-dispatcher-windows-386.tar.gz
  13199043  2021-03-29T21:01:39Z  gs://kubectl-dispatcher/v1.17.17/release/kubectl-dispatcher-windows-amd64.tar.gz
TOTAL: 6 objects, 78693247 bytes (75.05 MiB)
```

TODO(seans): Update permissions on this bucket to allow other Googlers to write
to it.

## Packaging

From the `kubectl-dispatcher` artifacts and the official `kubectl` binaries
released by the Kubernetes project, we package a set of binaries for each
supported os/architecture combination. From the perforce client root directory:

```bash
$ ./third_party/hosted_kubernetes_e2e/scripts/compile_clients.py 1.17.17
```

The range of delegate `kubectl` versions supported is currently hard-coded in
the `compile-clients.py` script (these delegate versions should become
parameters). In this case, we support versions 1.16 through 1.20, and we
determined the range of delegates by querying the range of valid GKE cluster
versions:

```bash
$ gcloud container get-server-config --zone=us-west1
Fetching server config for us-west1
channels:
- channel: RAPID
  defaultVersion: 1.19.8-gke.1000
  validVersions:
  - 1.20.4-gke.1800
  - 1.19.8-gke.1600
  - 1.19.8-gke.1000
- channel: REGULAR
  defaultVersion: 1.18.15-gke.1501
  validVersions:
  - 1.18.16-gke.302
  - 1.18.15-gke.1502
  - 1.18.15-gke.1501
- channel: STABLE
  defaultVersion: 1.17.17-gke.1101
  validVersions:
  - 1.17.17-gke.2800
  - 1.17.17-gke.1101
  - 1.16.15-gke.7801
defaultClusterVersion: 1.17.17-gke.1101
```

An example CL to package `kubectl` into gcloud:

[Updates default kubectl from 1.16 to 1.17](https://critique-ng.corp.google.com/cl/345293441)

## Downloading kubectl through gcloud

```bash
$ gcloud components update kubectl
```

Once all gcloud `kubectl` artifacts are downloaded it will look like:

```bash
$ ls -l ~/google-cloud-sdk/bin/kubectl*
-rwxr-xr-x 1 seans primarygroup 43479040 Jan  1  1980 /usr/local/google/home/seans/google-cloud-sdk/bin/kubectl
-rwxr-xr-x 1 seans primarygroup 42967040 Jan  1  1980 /usr/local/google/home/seans/google-cloud-sdk/bin/kubectl.1.16
-rwxr-xr-x 1 seans primarygroup 43458560 Jan  1  1980 /usr/local/google/home/seans/google-cloud-sdk/bin/kubectl.1.17
-rwxr-xr-x 1 seans primarygroup 43986944 Jan  1  1980 /usr/local/google/home/seans/google-cloud-sdk/bin/kubectl.1.18
-rwxr-xr-x 1 seans primarygroup 42950656 Jan  1  1980 /usr/local/google/home/seans/google-cloud-sdk/bin/kubectl.1.19
-rwxr-xr-x 1 seans primarygroup 43059232 Jan  1  1980 /usr/local/google/home/seans/google-cloud-sdk/bin/kubectl.1.20
```

## Troubleshooting

TODO(seans): Describe compile errors due to wrong CROSS_VERSION. Current
CROSS_VERSION=v1.13.9-5 for 1.17 & 1.18.
