# Kubernetes Minion Charm

[Kubernetes](https://github.com/googlecloudplatform/kubernetes) is an open
source  system for managing containerized applications across multiple hosts.
Kubernetes uses [Docker](http://www.docker.io/) to package, instantiate and run
containerized applications.

The Kubernetes Juju charms enable you to run Kubernetes on all the cloud
platforms that Juju supports.

A Kubernetes deployment consists of several independent charms that can be
scaled to meet your needs

### Etcd
Etcd is a key value store for Kubernetes.  All persistent master state
is stored in `etcd`.

### Flannel-docker
Flannel is a
[software defined networking](http://en.wikipedia.org/wiki/Software-defined_networking)
component that provides individual subnets for each machine in the cluster.

### Docker
Docker is an open platform for distributing applications for system administrators.

### Kubernetes master
The controlling unit in a Kubernetes cluster is called the master.  It is the
main management contact point providing many management services for the worker
nodes.

### Kubernetes minion
The servers that perform the work are known as minions.  Minions must be able to
communicate with the master and run the workloads that are assigned to them.


## Usage

#### Deploying the Development Focus

To deploy a Kubernetes environment in Juju :

    juju deploy cs:~kubernetes/trusty/etcd
    juju deploy cs:trusty/flannel-docker
    juju deploy cs:trusty/docker
    juju deploy local:trusty/kubernetes-master
    juju deploy local:trusty/kubernetes

    juju add-relation etcd flannel-docker
    juju add-relation flannel-docker:network docker:network
    juju add-relation flannel-docker:docker-host docker
    juju add-relation etcd kubernetes
    juju add-relation etcd kubernetes-master
    juju add-relation kubernetes kubernetes-master


#### Deploying the recommended configuration

A bundle can be used to deploy Kubernetes onto any cloud it can be
orchestrated directly in the Juju Graphical User Interface, when using
`juju quickstart`:

    juju quickstart https://raw.githubusercontent.com/whitmo/bundle-kubernetes/master/bundles.yaml


For more information on the recommended bundle deployment, see the
[Kubernetes bundle documentation](https://github.com/whitmo/bundle-kubernetes)


#### Post Deployment

To interact with the kubernetes environment, either build or
[download](https://github.com/GoogleCloudPlatform/kubernetes/releases) the
[kubectl](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/kubectl.md)
binary (available in the releases binary tarball) and point it to the master with :


    $ juju status kubernetes-master | grep public
    public-address: 104.131.108.99
    $ export KUBERNETES_MASTER="104.131.108.99"

# Configuration
For you convenience this charm supports changing the version of kubernetes binaries.
This can be done through the Juju GUI or on the command line:

    juju set kubernetes version=”v0.10.0”

If the charm does not already contain the tar file with the desired architecture
and version it will attempt to download the kubernetes binaries using the gsutil
command.

Congratulations you know have deployed a Kubernetes environment! Use the
[kubectl](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/kubectl.md)
to interact with the environment.

# Kubernetes information

- [Kubernetes github project](https://github.com/GoogleCloudPlatform/kubernetes)
- [Kubernetes issue tracker](https://github.com/GoogleCloudPlatform/kubernetes/issues)
- [Kubernetes Documenation](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/docs)
- [Kubernetes releases](https://github.com/GoogleCloudPlatform/kubernetes/releases)
