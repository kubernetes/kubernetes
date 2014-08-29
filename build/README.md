# Building Kubernetes

To build Kubernetes you need to have access to a Docker installation through either of the following methods:

## Requirements

1. Be running Docker.  2 options supported/tested:
  1. **Mac OS X** The best way to go is to use `boot2docker`.  See instructions [here](https://docs.docker.com/installation/mac/).
  2. **Linux with local Docker**  Install Docker according to the [instructions](https://docs.docker.com/installation/#installation) for your OS.  The scripts here assume that they are using a local Docker server and that they can "reach around" docker and grab results directly from the file system.
2. Have python installed.  Pretty much it is installed everywhere at this point so you can probably ignore this.
3. For releasing, have the [Google Cloud SDK](https://developers.google.com/cloud/sdk/) installed and configured.  The default release mechanism will upload Docker images to a private registry backed by Google Cloud Storage.  Non-image release artifacts will be uploaded to Google Cloud Storage also.

## Key scripts

* `make-binaries.sh`: This will compile all of the Kubernetes binaries in a Docker container
* `run-tests.sh`: This will run the Kubernetes unit tests in a Docker container
* `run-integration.sh`: This will build and run the integration test in a Docker container
* `make-cross.sh`: This will make all cross-compiled binaries (currently just kubecfg).
* `copy-output.sh`: This will copy the contents of `_output/build` from any remote Docker container to the local `_output/build`.  Right now this is only necessary on Mac OS X with `boot2docker`.
* `make-clean.sh`: Clean out the contents of `_output/build`.
* `shell.sh`: Drop into a `bash` shell in a build container with a snapshot of the current repo code.
* `release.sh`: Build everything, test it, upload the results to a GCS bucket.  Docker images are also sent to the same bucket using the [`google/docker-registry`](https://registry.hub.docker.com/u/google/docker-registry/) Docker image.

## Releasing

The `release.sh` script will build a release.  It will build binaries, run tests, build runtime Docker images and then upload all build artifacts to a GCS bucket.

The GCS bucket that is used is named `kubernetes-releases-NNNNN`.  The `NNNNN` is a random string derived from an md5 hash of the project name.

The release process can be customized with environment variables:
* `KUBE_RELEASE_BUCKET`: Override the bucket to be used for uploading releases.
* `KUBE_RELEASE_PREFIX`: The prefix for all non-docker image build artifacts.  This defaults to `devel/`
* `KUBE_DOCKER_REG_PREFIX`: The prefix for storage of the docker registry.  This defaults to `docker-reg/`

The release Docker images (all defined in `build/run-images/*/Dockerfile`):
* `kubernetes-apiserver`: Runs the main API server. It is parameterized with environment variables for `ETCD_SERVERS` and `KUBE_MINIONS` with defaults for localhost.
* `kubernetes-controller-manager`: Runs a set external controllers (see `DESIGN.md` for details).  It is parameterized with environment variables for `ETCD_SERVERS` and `API_SERVER`.
* `kubernetes-proxy`: Runs the proxy server on each individual node.  This is parameterized for `ETCD_SERVERS` and is required to be launched with `--net=host` Docker option to function correctly.

Other build artifacts:
* **TODO:** package up client utilties and cluster bring up scripts.

## Basic Flow

The scripts directly under `build/` are used to build and test.  They will ensure that the `kube-build` Docker image is built (based on `build/build-image/Dockerfile`) and then execute the appropriate command in that container.  If necessary (for Mac OS X), the scripts will also copy results out.

The `kube-build` container image is built by first creating a "context" directory in `_output/images/build-image`.  It is done there instead of at the root of the Kubernetes repo to minimize the amount of data we need to package up when building the image.

Everything in `build/build-image/` is meant to be run inside of the container.  If it doesn't think it is running in the container it'll throw a warning.  While you can run some of that stuff outside of the container, it wasn't built to do so.

The files necessarily for the release Docker images are in `build/run-images/*`.  All of this is staged into `_output/images` similar to build-image.  The `base` image is used as a base for each of the specialized containers and is generally never pushed to a shared repository.

## TODOs

@jbeda is going on vacation and can't complete this work for a while.  Here are the logical next steps:

* [ ] Get a cluster up and running with the Docker images.  Perhaps start with a local cluster and move up to a GCE cluster.
* [ ] Implement #186 and #187.  This will make it easier to develop Kubernetes.
* [ ] Deprecate/replace most of the stuff in the hack/
* [ ] Put together a true client distribution.  You should be able to download the tarball for your platform and get a Kube cluster running in <5 minutes.  Not `git clone`.
* [ ] Plumb in a version so we can do proper versioned releases.  Probably create a `./VERSION` and dope it with the git hash or something?
* [ ] Create an install script that'll let us do a `curl https://[URL] | bash` to get that tarball down and ensure that other dependencies (cloud SDK?) are installed and configured correctly.
* [ ] Support Windows as a client.
* [ ] Support uploading to the Docker index instead of the GCS bucket.  This'll allow easier installs for those not running on GCE
