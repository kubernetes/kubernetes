<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<h1>PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
# Kubernetes User Guide

The user guide is intended for anyone who wants to run programs and services
on an existing Kubernetes cluster.  Setup and administration of a
Kubernetes cluster is described in the [Cluster Admin Guide](../admin/README.md).
The [Developer Guide](../developer-guide.md) is for anyone wanting to either write code which directly accesses the
kubernetes API, or to contribute directly to the kubernetes project.

## Primary concepts

* **Overview** ([overview.md](overview.md)): A brief overview
  of Kubernetes concepts. 

* **Nodes** ([docs/admin/node.md](../admin/node.md)): A node is a worker machine in Kubernetes.

* **Pods** ([pods.md](pods.md)): A pod is a tightly-coupled group of containers
  with shared volumes.

* **The Life of a Pod** ([pod-states.md](pod-states.md)):
  Covers the intersection of pod states, the PodStatus type, the life-cycle
  of a pod, events, restart policies, and replication controllers.

* **Replication Controllers** ([replication-controller.md](replication-controller.md)):
  A replication controller ensures that a specified number of pod "replicas" are 
  running at any one time.

* **Services** ([services.md](services.md)): A Kubernetes service is an abstraction 
  which defines a logical set of pods and a policy by which to access them.

* **Volumes** ([volumes.md](volumes.md)): A Volume is a directory, possibly with some 
  data in it, which is accessible to a Container.

* **Labels** ([labels.md](labels.md)): Labels are key/value pairs that are 
  attached to objects, such as pods. Labels can be used to organize and to 
  select subsets of objects. 

* **Secrets** ([secrets.md](secrets.md)): A Secret stores sensitive data
  (e.g. ssh keys, passwords) separately from the Pods that use them, protecting
  the sensitive data from proliferation by tools that process pods.

* **Accessing the API and other cluster services via a Proxy** [accessing-the-cluster.md](accessing-the-cluster.md)

* **API Overview** ([docs/api.md](../api.md)): Pointers to API documentation on various topics
  and explanation of Kubernetes's approaches to API changes and API versioning.

* **Kubernetes Web Interface** ([ui.md](ui.md)): Accessing the Kubernetes
  web user interface.

* **Kubectl Command Line Interface** ([kubectl/kubectl.md](kubectl/kubectl.md)):
  The `kubectl` command line reference.

* **Sharing Cluster Access** ([sharing-clusters.md](sharing-clusters.md)):
  How to share client credentials for a kubernetes cluster.

* **Roadmap** ([docs/roadmap.md](../roadmap.md)): The set of supported use cases, features,
  docs, and patterns that are required before Kubernetes 1.0.

* **Glossary** ([docs/glossary.md](../glossary.md)): Terms and concepts.

## Further reading
<!--- make sure all documents from the docs directory are linked somewhere.
This one-liner (execute in docs/ dir) prints unlinked documents (only from this
dir - no recursion):
for i in *.md; do grep -r $i . | grep -v "^\./$i" > /dev/null; rv=$?; if [[ $rv -ne 0 ]]; then echo $i; fi; done
-->

* **Annotations** ([annotations.md](annotations.md)): Attaching
  arbitrary non-identifying metadata.

* **Downward API** ([downward-api.md](downward-api.md)): Accessing system
  configuration from a pod without accessing Kubernetes API (see also
  [container-environment.md](container-environment.md)).

* **Kubernetes Container Environment** ([container-environment.md](container-environment.md)):
  Describes the environment for Kubelet managed containers on a Kubernetes
  node (see also [downward-api.md](downward-api.md)).

* **DNS Integration with SkyDNS** ([docs/admin/dns.md](../admin/dns.md)):
  Resolving a DNS name directly to a Kubernetes service.

* **Identifiers** ([identifiers.md](identifiers.md)): Names and UIDs
  explained.

* **Images** ([images.md](images.md)): Information about container images
  and private registries.

* **Logging** ([logging.md](logging.md)): Pointers to logging info.

* **Namespaces** ([namespaces.md](namespaces.md)): Namespaces help different
  projects, teams, or customers to share a kubernetes cluster.

* **Networking** ([docs/admin/networking.md](../admin/networking.md)): Pod networking overview.

* **Services and firewalls** ([docs/services-firewalls.md](../services-firewalls.md)): How
  to use firewalls.

* **Compute Resources** ([compute-resources.md](compute-resources.md)):
  Provides resource information such as size, type, and quantity to assist in
  assigning Kubernetes resources appropriately.

* The [API object documentation](http://kubernetes.io/third_party/swagger-ui/).

* Frequently asked questions are answered on this project's [wiki](https://github.com/GoogleCloudPlatform/kubernetes/wiki).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/user-guide.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
