# Building Kubernetes

To build Kubernetes you need to have access to a Docker installation through either of the following methods:

## Requirements

1. Run on Mac OS X.  The best way to go is to use `boot2docker`.  See instructions [here](https://docs.docker.com/installation/mac/).
2. Run on Linux against a local Docker.  Install Docker according to the [instructions](https://docs.docker.com/installation/#installation) for your OS.  The scripts here assume that they are using a local Docker server and that they can "reach around" docker and grab results directly from the file system.

## Basic Flow

The scripts directly under `build/` are used to build and test.  They will ensure that the `kube-build` Docker image is built (based on `build/build-image/Dockerfile`) and then execute the appropriate command in that container.  If necessary (for Mac OS X), the scripts will also copy results out.

The `kube-build` container image is built by first creating a "context" directory in `output/build-image`.  It is done there instead of at the root of the Kubernetes repo to minimize the amount of data we need to package up when building the image.

Everything in `build/build-image/` is meant to be run inside of the container.  If it doesn't think it is running in the container it'll throw a warning.  While you can run some of that stuff outside of the container, it wasn't built to do so.
