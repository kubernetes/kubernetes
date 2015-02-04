# Kubernetes

[![GoDoc](https://godoc.org/github.com/GoogleCloudPlatform/kubernetes?status.png)](https://godoc.org/github.com/GoogleCloudPlatform/kubernetes) [![Travis](https://travis-ci.org/GoogleCloudPlatform/kubernetes.svg?branch=master)](https://travis-ci.org/GoogleCloudPlatform/kubernetes)

Kubernetes is an open source system for managing containerized applications across multiple hosts,
providing basic mechanisms for deployment, maintenance, and scaling of applications.

Kubernetes is:

* **lean**: lightweight, simple, accessible
* **portable**: public, private, hybrid, multi cloud
* **extensible**: modular, pluggable, hookable, composable
* **self-healing**: auto-placement, auto-restart, auto-replication

Kubernetes builds upon a decade and a half of experience at Google running production workloads at scale, combined with best-of-breed ideas and practices from the community.

<hr>

### Kubernetes can run anywhere!
However, initial development was done on GCE and so our instructions and scripts are built around that.  If you make it work on other infrastructure please let us know and contribute instructions/code.

### Kubernetes is in pre-production beta!
While the concepts and architecture in Kubernetes represent years of experience designing and building large scale cluster manager at Google, the Kubernetes project is still under heavy development.  Expect bugs, design and API changes as we bring it to a stable, production product over the coming year.


## Concepts

Kubernetes works with the following concepts:

**Clusters** are the compute resources on top of which your containers are built. Kubernetes can run anywhere! See the [Getting Started Guides](docs/getting-started-guides) for instructions for a variety of services.

**Pods** are a colocated group of Docker containers with shared volumes. They're the smallest deployable units that can be created, scheduled, and managed with Kubernetes. Pods can be created individually, but it's recommended that you use a replication controller even if creating a single pod. [More about pods](docs/pods.md).

**Replication controllers** manage the lifecycle of pods. They ensure that a specified number of pods are running
at any given time, by creating or killing pods as required. [More about replication controllers](docs/replication-controller.md).

**Services** provide a single, stable name and address for a set of pods.
They act as basic load balancers. [More about services](docs/services.md).

**Labels** are used to organize and select groups of objects based on key:value pairs. [More about labels](docs/labels.md).

## Documentation

Kubernetes documentation is organized into several categories.

  - **Getting Started Guides**
    - for people who want to create a kubernetes cluster
    - in [docs/getting-started-guides](docs/getting-started-guides)
  - **User Documentation**
    - [User FAQ](https://github.com/GoogleCloudPlatform/kubernetes/wiki/User-FAQ)
    - in [docs](docs/overview.md)
    - for people who want to run programs on kubernetes
    - describes current features of the system (with brief mentions of planned features)
  - **Developer Documentation**
    - in [docs/devel](docs/devel)
    - for people who want to contribute code to kubernetes
    - covers development conventions
    - explains current architecture and project plans
  - **Service Documentation**
    - [Service FAQ](https://github.com/GoogleCloudPlatform/kubernetes/wiki/Services-FAQ)
    - for people who are interested in how Services work
    - details of ```kube-proxy``` iptables
    - how to wire services to external internet
  - **Design Documentation**
    - in [docs/design](docs/design)
    - for people who want to understand the design choices made
    - describes tradeoffs, alternative designs
    - descriptions of planned features that are too long for a github issue.
  - **Walkthroughs and Examples**
    - in [examples](/examples)
    - Hands on introduction and example config files
  - **API documentation**
    - in [the API conventions doc](docs/api-conventions.md)
    - and automatically generated API documentation served by the master
  - **Wiki/FAQ**
    - in [wiki](https://github.com/GoogleCloudPlatform/kubernetes/wiki)
    - includes a number of [Kubernetes community-contributed recipes](/contrib/recipes)

## Community, discussion and support

If you have questions or want to start contributing please reach out.  We don't bite!

The Kubernetes team is hanging out on IRC on the [#google-containers channel on freenode.net](http://webchat.freenode.net/?channels=google-containers).  We also have the [google-containers Google Groups mailing list](https://groups.google.com/forum/#!forum/google-containers) for questions and discussion as well as the [kubernetes-announce mailing list](https://groups.google.com/forum/#!forum/kubernetes-announce) for important announcements (low-traffic, no chatter).

If you are a company and are looking for a more formal engagement with Google around Kubernetes and containers at Google as a whole, please fill out [this form](https://docs.google.com/a/google.com/forms/d/1_RfwC8LZU4CKe4vKq32x5xpEJI5QZ-j0ShGmZVv9cm4/viewform) and we'll be in touch.

