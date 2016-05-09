# Kubernetes
FIXME

[![GoReportCard Widget]][GoReportCard] [![GoDoc Widget]][GoDoc] [![Travis Widget]][Travis] [![Coverage Status Widget]][Coverage Status]

[GoDoc]: https://godoc.org/k8s.io/kubernetes
[GoDoc Widget]: https://godoc.org/k8s.io/kubernetes?status.svg
[GoReportCard]: https://goreportcard.com/report/k8s.io/kubernetes
[GoReportCard Widget]: https://goreportcard.com/badge/k8s.io/kubernetes
[Travis]: https://travis-ci.org/kubernetes/kubernetes
[Travis Widget]: https://travis-ci.org/kubernetes/kubernetes.svg?branch=master
[Coverage Status]: https://coveralls.io/r/kubernetes/kubernetes
[Coverage Status Widget]: https://coveralls.io/repos/kubernetes/kubernetes/badge.svg

### Are you ...

  * Interested in learning more about using Kubernetes?  Please see our user-facing documentation on [kubernetes.io](http://kubernetes.io)
  * Interested in hacking on the core Kubernetes code base?  Keep reading!

<hr>

Kubernetes is an open source system for managing [containerized applications](https://github.com/kubernetes/kubernetes/wiki/Why-Kubernetes%3F#why-containers) across multiple hosts,
providing basic mechanisms for deployment, maintenance, and scaling of applications.

Kubernetes is:

* **lean**: lightweight, simple, accessible
* **portable**: public, private, hybrid, multi cloud
* **extensible**: modular, pluggable, hookable, composable
* **self-healing**: auto-placement, auto-restart, auto-replication

Kubernetes builds upon a [decade and a half of experience at Google running production workloads at scale](https://research.google.com/pubs/pub43438.html), combined with best-of-breed ideas and practices from the community.

<hr>

### Kubernetes can run anywhere!

However, initial development was done on GCE and so our instructions and scripts are built around that.  If you make it work on other infrastructure please let us know and contribute instructions/code.

### Kubernetes is ready for Production!

With the [1.0.1 release](https://github.com/kubernetes/kubernetes/releases/tag/v1.0.1) Kubernetes is ready to serve your production workloads.


## Concepts

Kubernetes works with the following concepts:

[**Cluster**](docs/admin/README.md)
: A cluster is a set of physical or virtual machines and other infrastructure resources used by Kubernetes to run your applications. Kubernetes can run anywhere! See the [Getting Started Guides](docs/getting-started-guides/) for instructions for a variety of services.

[**Node**](docs/admin/node.md)
: A node is a physical or virtual machine running Kubernetes, onto which pods can be scheduled.

[**Pod**](docs/user-guide/pods.md)
: Pods are a colocated group of application containers with shared volumes. They're the smallest deployable units that can be created, scheduled, and managed with Kubernetes. Pods can be created individually, but it's recommended that you use a replication controller even if creating a single pod.

[**Replication controller**](docs/user-guide/replication-controller.md)
: Replication controllers manage the lifecycle of pods. They ensure that a specified number of pods are running
at any given time, by creating or killing pods as required.

[**Service**](docs/user-guide/services.md)
: Services provide a single, stable name and address for a set of pods.
They act as basic load balancers.

[**Label**](docs/user-guide/labels.md)
: Labels are used to organize and select groups of objects based on key:value pairs.

## Documentation

Kubernetes documentation is organized into several categories.

  - **Getting started guides**
    - for people who want to create a Kubernetes cluster
      - in [Creating a Kubernetes Cluster](docs/getting-started-guides/README.md)
    - for people who want to port Kubernetes to a new environment
      - in [Getting Started from Scratch](docs/getting-started-guides/scratch.md)
  - **User documentation**
    - for people who want to run programs on an existing Kubernetes cluster
    - in the [Kubernetes User Guide: Managing Applications](docs/user-guide/README.md)
	*Tip: You can also view help documentation out on [http://kubernetes.io/docs/](http://kubernetes.io/docs/).*
    - the [Kubectl Command Line Interface](docs/user-guide/kubectl/kubectl.md) is a detailed reference on
      the `kubectl` CLI
    - [User FAQ](https://github.com/kubernetes/kubernetes/wiki/User-FAQ)
  - **Cluster administrator documentation**
    - for people who want to create a Kubernetes cluster and administer it
    - in the [Kubernetes Cluster Admin Guide](docs/admin/README.md)
  - **Developer and API documentation**
    - for people who want to write programs that access the Kubernetes API, write plugins
      or extensions, or modify the core Kubernetes code
    - in the [Kubernetes Developer Guide](docs/devel/README.md)
    - see also [notes on the API](docs/api.md)
    - see also the [API object documentation](docs/api-reference/README.md), a
      detailed description of all fields found in the core API objects
  - **Walkthroughs and examples**
    - hands-on introduction and example config files
    - in the [user guide](docs/user-guide/README.md#quick-walkthrough)
    - in the [docs/examples directory](examples/)
  - **Contributions from the Kubernetes community**
    - in the [docs/contrib directory](contrib/)
  - **Design documentation and design proposals**
    - for people who want to understand the design of Kubernetes, and feature proposals
    - design docs in the [Kubernetes Design Overview](docs/design/README.md) and the [docs/design directory](docs/design/)
    - proposals in the [docs/proposals directory](docs/proposals/)
  - **Wiki/FAQ**
    - in the [wiki](https://github.com/kubernetes/kubernetes/wiki)
    - troubleshooting information in the [troubleshooting guide](docs/troubleshooting.md)

## Community, discussion, contribution, and support

See which companies are committed to driving quality in Kubernetes on our [community page](http://kubernetes.io/community/).

Do you want to help "shape the evolution of technologies that are container packaged, dynamically scheduled and microservices oriented?"

You should consider joining the [Cloud Native Computing Foundation](https://cncf.io/about). For details about who's involved and how Kubernetes plays a role, read [their announcement](https://cncf.io/news/announcement/2015/07/new-cloud-native-computing-foundation-drive-alignment-among-container).

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).

### Are you ready to add to the discussion?

We have presence on:

 * [Twitter](https://twitter.com/kubernetesio)
 * [Google+](https://plus.google.com/u/0/b/116512812300813784482/116512812300813784482)
 * [Blogger](http://blog.kubernetes.io/)

You can also view recordings of past events and presentations on our [Media page](http://kubernetes.io/media/).

For Q&A, our threads are at:

 * [Stack Overflow](http://stackoverflow.com/questions/tagged/kubernetes)
 * [Slack](http://slack.k8s.io/)

### Want to do more than just 'discuss' Kubernetes?

If you're interested in being a contributor and want to get involved in developing Kubernetes, start in the [Kubernetes Developer Guide](docs/devel/README.md) and also review the [contributor guidelines](CONTRIBUTING.md).

### Support

While there are many different channels that you can use to get ahold of us, you can help make sure that we are efficient in getting you the help that you need.

If you need support, start with the [troubleshooting guide](docs/troubleshooting.md#getting-help) and work your way through the process that we've outlined.

That said, if you have questions, reach out to us one way or another.  We don't bite!

### Community resources:

* **Awesome-kubernetes**:

You can find more projects, tools and articles related to Kubernetes on the  [awesome-kubernetes](https://github.com/ramitsurana/awesome-kubernetes) list. Add your project there and help us make it better.

* **CoreKube** - [https://corekube.com](https://corekube.com):

Instructive & educational resources for the Kubernetes community. By the community.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/README.md?pixel)]()
