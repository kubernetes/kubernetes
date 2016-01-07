# Kubernetes Master Charm

[Kubernetes](https://github.com/kubernetes/kubernetes) is an open
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

Use the 'juju quickstart' command to deploy a Kubernetes cluster to any cloud
supported by Juju.  

The charm store version of the Kubernetes bundle can be deployed as follows:

    juju quickstart u/kubernetes/kubernetes-cluster

> Note: The charm store bundle may be locked to a specific Kubernetes release.

Alternately you could deploy a Kubernetes bundle straight from github or a file:

    juju quickstart https://raw.githubusercontent.com/kubernetes/kubernetes/master/cluster/juju/bundles/local.yaml

The command above does few things for you:

- Starts a curses based gui for managing your cloud or MAAS credentials
- Looks for a bootstrapped deployment environment, and bootstraps if
  required. This will launch a bootstrap node in your chosen
  deployment environment (machine 0).
- Deploys the Juju GUI to your environment onto the bootstrap node.
- Provisions 4 machines, and deploys the Kubernetes services on top of
  them (Kubernetes-master, two Kubernetes minions using flannel, and etcd).
- Orchestrates the relations among the services, and exits.

Now you should have a running Kubernetes. Run `juju status
--format=oneline` to see the address of your kubernetes-master unit.

For further reading on [Juju Quickstart](https://pypi.python.org/pypi/juju-quickstart)

Go to the [Getting started with Juju guide](https://github.com/kubernetes/kubernetes/blob/master/docs/getting-started-guides/juju.md)
for more information about deploying a development Kubernetes cluster.


#### Post Deployment

To interact with the kubernetes environment, either build or
[download](https://github.com/kubernetes/kubernetes/releases) the
[kubectl](https://github.com/kubernetes/kubernetes/blob/master/docs/user-guide/kubectl/kubectl.md)
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
[kubectl](https://github.com/kubernetes/kubernetes/blob/master/docs/user-guide/kubectl/kubectl.md)
to interact with the environment.

# Kubernetes information

- [Kubernetes github project](https://github.com/kubernetes/kubernetes)
- [Kubernetes issue tracker](https://github.com/kubernetes/kubernetes/issues)
- [Kubernetes Documenation](https://github.com/kubernetes/kubernetes/tree/master/docs)
- [Kubernetes releases](https://github.com/kubernetes/kubernetes/releases)


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/juju/charms/trusty/kubernetes-master/README.md?pixel)]()
