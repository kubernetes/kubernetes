# kubernetes-bundle

The kubernetes-bundle allows you to deploy the many services of
Kubernetes to a cloud environment and get started using the Kubernetes
technology quickly.

## Kubernetes

Kubernetes is an open source system for managing containerized
applications.  Kubernetes uses [Docker](http://docker.com) to run
containerized applications.

## Juju TL;DR

The [Juju](https://juju.ubuntu.com) system provides provisioning and
orchestration across a variety of clouds and bare metal. A juju bundle
describes collection of services and how they interelate. `juju
quickstart` allows you to bootstrap a deployment environment and
deploy a bundle.

## Dive in!

#### Install Juju Quickstart

You will need to
[install the Juju client](https://juju.ubuntu.com/install/) and
`juju-quickstart` as pre-requisites.  To deploy the bundle use
`juju-quickstart` which runs on Mac OS (`brew install
juju-quickstart`) or Ubuntu (`apt-get install juju-quickstart`).

### Deploy Kubernetes Bundle

Deploy Kubernetes onto any cloud and orchestrated directly in the Juju
Graphical User Interface using `juju quickstart`:

    juju quickstart -i https://raw.githubusercontent.com/whitmo/bundle-kubernetes/master/bundles.yaml

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
--format=oneline` to see the address of your kubernetes master.

For further reading on [Juju Quickstart](https://pypi.python.org/pypi/juju-quickstart)

### Using the Kubernetes Client

You'll need the Kubernetes command line client,
[kubectl](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/kubectl.md)
to interact with the created cluster.  The kubectl command is
installed on the kubernetes-master charm. If you want to work with
the cluster from your computer you will need to install the binary
locally (see instructions below).

You can access kubectl by a number ways using juju.

via juju run:

    juju run --service kubernetes-master/0 "sudo kubectl get mi"

via juju ssh:

    juju ssh kubernetes-master/0 -t "sudo kubectl get mi"

You may also `juju ssh kubernetes-master/0` and call kubectl from that
machine.

See the
[kubectl documentation](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/kubectl.md)
for more details of what can be done with the command line tool.

### Scaling up the cluster

You can add capacity by adding more Docker units:

     juju add-unit docker

### Known Limitations

Kubernetes currently has several platform specific functionality. For
example load balancers and persistence volumes only work with the
Google Compute provider at this time.

The Juju integration uses the Kubernetes null provider. This means
external load balancers and storage can't be directly driven through
Kubernetes config files at this time. We look forward to adding these
capabilities to the charms.


## More about the components the bundle deploys

### Kubernetes master

The master controls the Kubernetes cluster.  It manages for the worker
nodes and provides the primary interface for control by the user.

### Kubernetes minion

The minions are the servers that perform the work.  Minions must
communicate with the master and run the workloads that are assigned to
them.

### Flannel-docker

Flannel provides individual subnets for each machine in the cluster by
creating a
[software defined networking](http://en.wikipedia.org/wiki/Software-defined_networking).

### Docker

An open platform for distributed applications for developers and sysadmins.

### Etcd

Etcd persists state for Flannel and Kubernetes. It is a distributed
key-value store with an http interface.


## For further information on getting started with Juju

Juju has complete documentation with regard to setup, and cloud
configuration on it's own
[documentation site](https://juju.ubuntu.com/docs/).

- [Getting Started](https://juju.ubuntu.com/docs/getting-started.html)
- [Using Juju](https://juju.ubuntu.com/docs/charms.html)


## Installing the kubectl outside of kubernetes master machine

Download the Kuberentes release from:
https://github.com/GoogleCloudPlatform/kubernetes/releases and extract
the release, you can then just directly use the cli binary at
./kubernetes/platforms/linux/amd64/kubectl

You'll need the address of the kubernetes-master as environment variable :

    juju status kubernetes-master/0

Grab the public-address there and export it as KUBERNETES_MASTER
environment variable :

    export KUBERNETES_MASTER=$(juju status --format=oneline kubernetes-master | cut -d' ' -f3):8080

And now you can run kubectl on the command line :

    kubectl get mi

See the
[kubectl documentation](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/kubectl.md)
for more details of what can be done with the command line tool.


## Hacking on the kubernetes-bundle and associated charms

The kubernetes-bundle is open source and available on github.com.  If
you want to get started developing on the bundle you can clone it from
github.  Often you will need the related charms which are also on
github.

    mkdir ~/bundles
    git clone https://github.com/whitmo/kubernetes-bundle.git ~/bundles/kubernetes-bundle
    mkdir -p ~/charms/trusty
    git clone https://github.com/whitmo/kubernetes-charm.git ~/charms/trusty/kubernetes
    git clone https://github.com/whitmo/kubernetes-master-charm.git ~/charms/trusty/kubernetes-master

    juju quickstart specs/develop.yaml

## How to contribute

Send us pull requests!  We'll send you a cookie if they include tests and docs.


## Current and Most Complete Information

 - [kubernetes-master charm on Github](https://github.com/whitmo/charm-kubernetes-master)
 - [kubernetes charm on GitHub](https://github.com/whitmo/charm-kubernetes)
 - [etcd charm on GitHub](https://github.com/whitmo/etcd-charm)
 - [Flannel charm on GitHub](https://github.com/chuckbutler/docker-flannel-charm)
 - [Docker charm on GitHub](https://github.com/chuckbutler/docker-charm)

More information about the
[Kubernetes project](https://github.com/GoogleCloudPlatform/kubernetes)
or check out the
[Kubernetes Documentation](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/docs)
for more details about the Kubernetes concepts and terminology.

Having a problem? Check the [Kubernetes issues database](https://github.com/GoogleCloudPlatform/kubernetes/issues)
for related issues.
