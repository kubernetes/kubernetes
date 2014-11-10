# Kubernetes
Kubernetes is an open source implementation of container cluster management.

[Kubernetes Design Document](DESIGN.md) - [Kubernetes @ Google I/O 2014](http://youtu.be/tsk0pWf4ipw)

[![GoDoc](https://godoc.org/github.com/GoogleCloudPlatform/kubernetes?status.png)](https://godoc.org/github.com/GoogleCloudPlatform/kubernetes)
[![Travis](https://travis-ci.org/GoogleCloudPlatform/kubernetes.svg?branch=master)](https://travis-ci.org/GoogleCloudPlatform/kubernetes)


## Kubernetes can run anywhere!
However, initial development was done on GCE and so our instructions and scripts are built around that.  If you make it work on other infrastructure please let us know and contribute instructions/code.

## Kubernetes is in pre-production beta!
While the concepts and architecture in Kubernetes represent years of experience designing and building large scale cluster manager at Google, the Kubernetes project is still under heavy development.  Expect bugs, design and API changes as we bring it to a stable, production product over the coming year.

### Contents
* Getting Started Guides
  * [Google Compute Engine](docs/getting-started-guides/gce.md)
  * [Locally](docs/getting-started-guides/locally.md)
  * [Vagrant](docs/getting-started-guides/vagrant.md)
  * [AWS with CoreOS and Cloud Formation](docs/getting-started-guides/aws-coreos.md)
  * [AWS](docs/getting-started-guides/aws.md)
  * Fedora (w/ [Ansible](docs/getting-started-guides/fedora/fedora_ansible_config.md) or [manual](docs/getting-started-guides/fedora/fedora_manual_config.md))
  * [Circle CI](https://circleci.com/docs/docker#google-compute-engine-and-kubernetes)
  * [Digital Ocean](https://github.com/bketelsen/coreos-kubernetes-digitalocean)
  * [CoreOS](docs/getting-started-guides/coreos.md)
  * [OpenStack](https://developer.rackspace.com/blog/running-coreos-and-kubernetes/)
  * [CloudStack](docs/getting-started-guides/cloudstack.md)
  * [Rackspace](docs/getting-started-guides/rackspace.md)
  * [vSphere](docs/getting-started-guides/vsphere.md)

* The following clouds are currently broken at Kubernetes head.  Please sync your client to `v0.3` (`git checkout v0.3`) to use these:
  * [Microsoft Azure](docs/getting-started-guides/azure.md)

* [Kubernetes 101](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/examples/walkthrough)
* [kubecfg command line tool](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/cli.md)
* [Kubernetes API Documentation](http://cdn.rawgit.com/GoogleCloudPlatform/kubernetes/31a0daae3627c91bc96e1f02a6344cd76e294791/api/kubernetes.html)
* [Kubernetes Client Libraries](docs/client-libraries.md)
* [Discussion and Community Support](#community-discussion-and-support)
* [Hacking on Kubernetes](CONTRIBUTING.md)
* [Hacking on Kubernetes Salt configuration](docs/salt.md)
* [Kubernetes User Interface](docs/ux.md)

## Where to go next?

Check out examples of Kubernetes in action, and community projects in the larger ecosystem:

* [Kubernetes 101](examples/walkthrough/README.md)
* [Kubernetes 201](examples/walkthrough/k8s201.md)
* [Detailed example application](examples/guestbook/README.md)
* [Example of dynamic updates](examples/update-demo/README.md)
* [Cluster monitoring with heapster and cAdvisor](https://github.com/GoogleCloudPlatform/heapster)
* [Community projects](https://github.com/GoogleCloudPlatform/kubernetes/wiki/Kubernetes-Community)
* [Development guide](docs/devel/development.md)
* [User contributed recipes](contrib/recipes)

Or fork and start hacking!

## Community, discussion and support

If you have questions or want to start contributing please reach out.  We don't bite!

The Kubernetes team is hanging out on IRC on the [#google-containers channel on freenode.net](http://webchat.freenode.net/?channels=google-containers).  We also have the [google-containers Google Groups mailing list](https://groups.google.com/forum/#!forum/google-containers) for questions and discussion as well as the [kubernetes-announce mailing list](https://groups.google.com/forum/#!forum/kubernetes-announce) for important announcements (low-traffic, no chatter).

If you are a company and are looking for a more formal engagement with Google around Kubernetes and containers at Google as a whole, please fill out [this form](https://docs.google.com/a/google.com/forms/d/1_RfwC8LZU4CKe4vKq32x5xpEJI5QZ-j0ShGmZVv9cm4/viewform) and we'll be in touch.
