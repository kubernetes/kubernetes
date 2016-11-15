# Kubernetes

[![Submit Queue Widget]][Submit Queue] [![GoDoc Widget]][GoDoc] [![Coverage Status Widget]][Coverage Status]

<img src="https://github.com/kubernetes/kubernetes/raw/master/logo/logo.png" width="100">

[Submit Queue]: http://submit-queue.k8s.io/#/e2e
[Submit Queue Widget]: http://submit-queue.k8s.io/health.svg?v=1
[GoDoc]: https://godoc.org/k8s.io/kubernetes
[GoDoc Widget]: https://godoc.org/k8s.io/kubernetes?status.svg
[Coverage Status]: https://coveralls.io/r/kubernetes/kubernetes
[Coverage Status Widget]: https://coveralls.io/repos/kubernetes/kubernetes/badge.svg

## Introduction

Kubernetes is an open source system for managing [containerized applications](http://kubernetes.io/docs/whatisk8s/) across multiple hosts,
providing basic mechanisms for deployment, maintenance, and scaling of applications. Kubernetes is hosted by the Cloud Native Computing Foundation ([CNCF](https://www.cncf.io))

Kubernetes is:

* **lean**: lightweight, simple, accessible
* **portable**: public, private, hybrid, multi cloud
* **extensible**: modular, pluggable, hookable, composable
* **self-healing**: auto-placement, auto-restart, auto-replication

Kubernetes builds upon a decade and a half of experience at Google running production workloads at scale using a system called [Borg](https://research.google.com/pubs/pub43438.html), combined with best-of-breed ideas and practices from the community.

<hr>

### Kubernetes is ready for Production !

Since the Kubernetes 1.0 release in July 2015 Kubernetes is ready for your production workloads. Check the [case studies](http://kubernetes.io/case-studies/).

### Kubernetes can run anywhere !

You can run Kubernetes on your local workstation, cloud providers (e.g. GCE, AWS, Azure), on-premises virtual machines and physical hardware. Essentially, anywhere Linux runs you can run Kubernetes. Checkout the [deployment solutions](http://kubernetes.io/docs/getting-started-guides/) for details.

<hr>

### Are you ...

  * Interested in learning more about using Kubernetes?  Please see our user-facing documentation on [kubernetes.io](http://kubernetes.io). Try our [interactive tutorial](http://kubernetes.io/docs/tutorials/kubernetes-basics/) or take a free course on [Scalable Microservices with Kubernetes](https://www.udacity.com/course/scalable-microservices-with-kubernetes--ud615).
  * Interested in hacking on the core Kubernetes code base, developing tools using the Kubernetes API or helping in anyway possible ?  Keep reading!

## Code of Conduct

The Kubernetes community abides by the CNCF [code of conduct](https://github.com/cncf/foundation/blob/master/code-of-conduct.md). Here is an excerpt:

_As contributors and maintainers of this project, and in the interest of fostering an open and welcoming community, we pledge to respect all people who contribute through reporting issues, posting feature requests, updating documentation, submitting pull requests or patches, and other activities._

## Concepts Overview

Kubernetes works with the following concepts:

[**Cluster**](http://kubernetes.io/docs/admin/)
: A cluster is a set of physical or virtual machines and other infrastructure resources used by Kubernetes to run your applications.

[**Node**](http://kubernetes.io/docs/admin/node/)
: A node is a physical or virtual machine running Kubernetes, onto which pods can be scheduled.

[**Pod**](http://kubernetes.io/docs/user-guide/pods/)
: Pods are a colocated group of application containers with shared volumes. They're the smallest deployable units that can be created, scheduled, and managed with Kubernetes. Pods can be created individually, but it's recommended that you use a replication controller even if creating a single pod.

[**Replication controller**](http://kubernetes.io/docs/user-guide/replication-controller/)
: Replication controllers manage the lifecycle of pods. They ensure that a specified number of pods are running
at any given time, by creating or killing pods as required.

[**Service**](http://kubernetes.io/docs/user-guide/services/)
: Services provide a single, stable name and address for a set of pods.
They act as basic load balancers.

[**Label**](http://kubernetes.io/docs/user-guide/labels/)
: Labels are used to organize and select groups of objects based on key:value pairs.

## Community

Do you want to help " shape the evolution of technologies that are container packaged, dynamically scheduled and microservices oriented? ". If you are a company, you should consider joining the [CNCF](https://cncf.io/about). For details about who's involved and how Kubernetes plays a role, read [the announcement](https://cncf.io/news/announcement/2015/07/new-cloud-native-computing-foundation-drive-alignment-among-container).

Join us on social media and read our blog:

 * [Twitter](https://twitter.com/kubernetesio)
 * [Google+](https://plus.google.com/u/0/b/116512812300813784482/116512812300813784482)
 * [Blog](http://blog.kubernetes.io/)

Ask questions and help answer them on:

 * [Stack Overflow](http://stackoverflow.com/questions/tagged/kubernetes)
 * [Slack](http://slack.k8s.io/)

Attend our key events:

* [kubecon](http://events.linuxfoundation.org/events/kubecon)
* [cloudnativecon](http://events.linuxfoundation.org/events/cloudnativecon)
* weekly [community meeting](https://github.com/kubernetes/community/blob/master/community/README.md)

Join a Special Interest Group ([SIG](https://github.com/kubernetes/community))

## Contribute

If you're interested in being a contributor and want to get involved in developing Kubernetes, get started with this light reading:

*  The community [expectations](docs/devel/community-expectations.md)
*  The [contributor guidelines](CONTRIBUTING.md)
*  The [Kubernetes Developer Guide](docs/devel/README.md)

You will then most certainly gain a lot from joining a [SIG](https://github.com/kubernetes/community), attending the regular hangouts as well as the community [meeting](https://github.com/kubernetes/community/blob/master/community/README.md).

If you have an idea for a new feature, see the [Kubernetes Features](https://github.com/kubernetes/features) repository for a list of features that are coming in new releases as well as details on how to propose one.

## Documentation

The Kubernetes [documentation](http://kubernetes.io/docs/) is organized into several categories.

  - **Getting started guides**
    - For people who want to create a Kubernetes cluster
      - See [Creating a Kubernetes Cluster](http://kubernetes.github.io/docs/getting-started-guides/)
    - For people who want to port Kubernetes to a new environment
      - See [Getting Started from Scratch](http://kubernetes.github.io/docs/getting-started-guides/scratch/)
  - **User documentation**
    - For people who want to run programs on an existing Kubernetes cluster
      - See the [Kubernetes User Guide: Managing Applications](http://kubernetes.github.io/docs/user-guide/)
  - **Administrator documentation**
    - For people who want to administer a Kubernetes cluster
      - See the [Kubernetes Cluster Admin Guide](http://kubernetes.io/docs/admin/)
  - **Developer and API documentation**
    - For people who want to write programs that access the Kubernetes [API](docs/api.md), write plugins
      or extensions, or modify the core Kubernetes code
      - See the [Kubernetes Developer Guide](docs/devel/README.md)
  - **Walkthroughs and examples**
    - For Hands-on introduction and example config files
      - See the [user guide](http://kubernetes.github.io/docs/user-guide/)
      - See the [examples](examples/) directory
  - **Contributions from the Kubernetes community**
    - See the [contrib](contrib/) repository
    - See the [incubator](https://github.com/kubernetes-incubator) organisation
  - **Design documentation and design proposals**
    - For people who want to understand the design of Kubernetes, and feature proposals
      - See the design docs in the [Kubernetes Design Overview](docs/design/README.md) and the [design](docs/design/) directory
      - See the proposals in the [proposals](docs/proposals/) directory
  - **Wiki/FAQ**
    - For general developer information see the [wiki](https://github.com/kubernetes/kubernetes/wiki)
    - For user and admin frequently asked questions, see the [troubleshooting guide](http://kubernetes.io/docs/troubleshooting/)

## Support

While there are many different channels that you can use to get hold of us ([Slack](http://slack.k8s.io/), [Stack Overflow](http://stackoverflow.com/questions/tagged/kubernetes), [Issues](https://github.com/kubernetes/kubernetes/issues/new), [Forums/Mailing lists](https://groups.google.com/forum/#!forum/kubernetes-users)), you can help make sure that we are efficient in getting you the help that you need.

If you need support, start with the [troubleshooting guide](http://kubernetes.io/docs/troubleshooting/) and work your way through the process that we've outlined.

That said, if you have questions, reach out to us one way or another.  We don't bite!

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/README.md?pixel)]()

