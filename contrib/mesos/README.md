# Kubernetes-Mesos

Kubernetes-Mesos modifies Kubernetes to act as an [Apache Mesos](http://mesos.apache.org/) framework.

## Features On Mesos

Kubernetes gains the following benefits when installed on Mesos:

- **Node-Level Auto-Scaling** - Kubernetes minion nodes are created automatically, up to the size of the provisioned Mesos cluster.
- **Resource Sharing** - Co-location of Kubernetes with other popular next-generation services on the same cluster (e.g. [Hadoop](https://github.com/mesos/hadoop), [Spark](http://spark.apache.org/), and [Chronos](https://mesos.github.io/chronos/), [Cassandra](http://mesosphere.github.io/cassandra-mesos/), etc.). Resources are allocated to the frameworks based on fairness and can be claimed or passed on depending on framework load.
- **Independence from special Network Infrastructure** - Mesos can (but of course doesn't have to) run on networks which cannot assign a routable IP to every container. The Kubernetes on Mesos endpoint controller is specially modified to allow pods to communicate with services in such an environment.

For more information about how Kubernetes-Mesos is different from Kubernetes, see [Architecture](./docs/architecture.md).


## Release Status

Kubernetes-Mesos is alpha quality, still under active development, and not yet recommended for production systems.

For more information about development progress, see the [known issues](./docs/issues.md) or the [kubernetes-mesos repository](https://github.com/mesosphere/kubernetes-mesos) where backlog issues are tracked.

## Usage

This project combines concepts and technologies from two already-complex projects: Mesos and Kubernetes. It may help to familiarize yourself with the basics of each project before reading on:

* [Mesos Documentation](http://mesos.apache.org/documentation/latest)
* [Kubernetes Documentation](../../README.md)

To get up and running with Kubernetes-Mesos, follow:

- the [Getting started guide](../../docs/getting-started-guides/mesos.md) to launch a Kubernetes-Mesos cluster,
- the [Kubernetes-Mesos Scheduler Guide](./docs/scheduler.md) for topics concerning the custom scheduler used in this distribution.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/mesos/README.md?pixel)]()
