## Getting started with Kubernetes on Mac OS X

The easiest way to run Kubernetes cluster on Mac OS X is with the help of Vagrant, which creates VirtualBox VMs. A set of platform native binaries are available to help manage clusters (look for the target `darwin`).

### Development setup

1. Install boot2docker from http://boot2docker.io
2. Follow the instructions for development under the `docs/devel` directory.

### Running cluster locally

The easiest way to run a cluster locally is via Vagrant. Follow the steps in the [Vagrant](vagrant.md) starter guide.

### Caveats

Be sure to have enough free disk and memory to run multiple VMs. The Vagrant steps will create at least 2 VM's, each with 512 MB of memory. It's recommended to have at least 2GB to 4GB of free memory (plus appropriate disk space).
