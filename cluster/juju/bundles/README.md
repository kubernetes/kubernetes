# kubernetes-bundle

The kubernetes-bundle allows you to deploy the many services of
Kubernetes to a cloud environment and get started using the Kubernetes
technology quickly.

## Kubernetes

Kubernetes is an open source system for managing containerized
applications.  Kubernetes uses [Docker](http://docker.com) to run
containerized applications.

## Juju TL;DR

The [Juju](https://jujucharms.com) system provides provisioning and
orchestration across a variety of clouds and bare metal. A juju bundle
describes collection of services and how they interrelate. `juju
quickstart` allows you to bootstrap a deployment environment and
deploy a bundle.

## Dive in!

#### Install Juju Quickstart

You will need to
[install the Juju client](https://jujucharms.com/get-started) and
`juju-quickstart` as prerequisites.  To deploy the bundle use
`juju-quickstart` which runs on Mac OS (`brew install
juju-quickstart`) or Ubuntu (`apt-get install juju-quickstart`).

### Deploy a Kubernetes Bundle

Use the 'juju quickstart' command to deploy a Kubernetes cluster to any cloud
supported by Juju.  

The charm store version of the Kubernetes bundle can be deployed as follows:

    juju quickstart u/kubernetes/kubernetes-cluster

> Note: The charm store bundle may be locked to a specific Kubernetes release.

Alternately you could deploy a Kubernetes bundle straight from github or a file:

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
--format=oneline` to see the address of your kubernetes-master unit.

For further reading on [Juju Quickstart](https://pypi.python.org/pypi/juju-quickstart)

Go to the [Getting started with Juju guide](https://kubernetes.io/docs/getting-started-guides/ubuntu/installation/#setting-up-kubernetes-with-juju)
for more information about deploying a development Kubernetes cluster.

### Using the Kubernetes Client

You'll need the Kubernetes command line client,
[kubectl](https://github.com/kubernetes/kubernetes/blob/master/docs/user-guide/kubectl/kubectl.md)
to interact with the created cluster.  The kubectl command is
installed on the kubernetes-master charm. If you want to work with
the cluster from your computer you will need to install the binary
locally.

You can access kubectl by a number ways using juju.

via juju run:

    juju run --service kubernetes-master/0 "sudo kubectl get nodes"

via juju ssh:

    juju ssh kubernetes-master/0 -t "sudo kubectl get nodes"

You may also SSH to the kubernetes-master unit (`juju ssh kubernetes-master/0`)
and call kubectl from the command prompt.

See the
[kubectl documentation](https://github.com/kubernetes/kubernetes/blob/master/docs/user-guide/kubectl/kubectl.md)
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
[documentation site](https://jujucharms.com/docs/).

- [Getting Started](https://jujucharms.com/docs/stable/getting-started)
- [Using Juju](https://jujucharms.com/docs/stable/charms)


## Installing the kubectl outside of kubernetes-master unit

Download the Kubernetes release from:
https://github.com/kubernetes/kubernetes/releases and extract
the release, you can then just directly use the cli binary at
./kubernetes/platforms/linux/amd64/kubectl

You'll need the address of the kubernetes-master as environment variable :

    juju status kubernetes-master/0

Grab the public-address there and export it as KUBERNETES_MASTER
environment variable :

    export KUBERNETES_MASTER=$(juju status --format=oneline kubernetes-master | grep kubernetes-master | cut -d' ' -f3):8080

And now you can run kubectl on the command line :

    kubectl get no

See the
[kubectl documentation](https://github.com/kubernetes/kubernetes/blob/master/docs/user-guide/kubectl/kubectl.md)
for more details of what can be done with the command line tool.


## Hacking on the kubernetes-bundle and associated charms

The kubernetes-bundle is open source and available on github.com.  If
you want to get started developing on the bundle you can clone it from
github.  

    git clone https://github.com/kubernetes/kubernetes.git

Go to the [Getting started with Juju guide](https://kubernetes.io/docs/getting-started-guides/ubuntu/installation/#setting-up-kubernetes-with-juju)
for more information about the bundle or charms.

## How to contribute

Send us pull requests!  We'll send you a cookie if they include tests and docs.


## Current and Most Complete Information

The charms and bundles are in the [kubernetes](https://github.com/kubernetes/kubernetes)
repository in github.

 - [kubernetes-master charm on GitHub](https://github.com/kubernetes/kubernetes/tree/master/cluster/juju/charms/trusty/kubernetes-master)
 - [kubernetes charm on GitHub](https://github.com/kubernetes/kubernetes/tree/master/cluster/juju/charms/trusty/kubernetes)


More information about the
[Kubernetes project](https://github.com/kubernetes/kubernetes)
or check out the
[Kubernetes Documentation](https://github.com/kubernetes/kubernetes/tree/master/docs)
for more details about the Kubernetes concepts and terminology.

Having a problem? Check the [Kubernetes issues database](https://github.com/kubernetes/kubernetes/issues)
for related issues.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/juju/bundles/README.md?pixel)]()
