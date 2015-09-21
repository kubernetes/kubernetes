<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes User Guide: Managing Applications

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Kubernetes User Guide: Managing Applications](#kubernetes-user-guide-managing-applications)
  - [Quick walkthrough](#quick-walkthrough)
  - [Thorough walkthrough](#thorough-walkthrough)
  - [Concept guide](#concept-guide)
  - [Further reading](#further-reading)

<!-- END MUNGE: GENERATED_TOC -->

The user guide is intended for anyone who wants to run programs and services on an existing Kubernetes cluster.  Setup and administration of a Kubernetes cluster is described in the [Cluster Admin Guide](../../docs/admin/README.md). The [Developer Guide](../../docs/devel/README.md) is for anyone wanting to either write code which directly accesses the Kubernetes API, or to contribute directly to the Kubernetes project.

Please ensure you have completed the [prerequisites for running examples from the user guide](prereqs.md).

## Quick walkthrough

1. [Kubernetes 101](walkthrough/README.md)
1. [Kubernetes 201](walkthrough/k8s201.md)

## Thorough walkthrough

If you don't have much familiarity with Kubernetes, we recommend you read the following sections in order:

1. [Quick start: launch and expose an application](quick-start.md)
1. [Configuring and launching containers: configuring common container parameters](configuring-containers.md)
1. [Deploying continuously running applications](deploying-applications.md)
1. [Connecting applications: exposing applications to clients and users](connecting-applications.md)
1. [Working with containers in production](production-pods.md)
1. [Managing deployments](managing-deployments.md)
1. [Application introspection and debugging](introspection-and-debugging.md)
    1. [Using the Kubernetes web user interface](ui.md)
    1. [Logging](logging.md)
    1. [Monitoring](monitoring.md)
    1. [Getting into containers via `exec`](getting-into-containers.md)
    1. [Connecting to containers via proxies](connecting-to-applications-proxy.md)
    1. [Connecting to containers via port forwarding](connecting-to-applications-port-forward.md)

## Concept guide

[**Overview**](overview.md)
: A brief overview of Kubernetes concepts.

[**Cluster**](../admin/README.md)
: A cluster is a set of physical or virtual machines and other infrastructure resources used by Kubernetes to run your applications.

[**Node**](../admin/node.md)
: A node is a physical or virtual machine running Kubernetes, onto which pods can be scheduled.

[**Pod**](pods.md)
: A pod is a co-located group of containers and volumes.

[**Label**](labels.md)
: A label is a key/value pair that is attached to a resource, such as a pod, to convey a user-defined identifying attribute. Labels can be used to organize and to select subsets of resources.

[**Selector**](labels.md#label-selectors)
: A selector is an expression that matches labels in order to identify related resources, such as which pods are targeted by a load-balanced service.

[**Replication Controller**](replication-controller.md)
: A replication controller ensures that a specified number of pod replicas are running at any one time. It both allows for easy scaling of replicated systems and handles re-creation of a pod when the machine it is on reboots or otherwise fails.

[**Service**](services.md)
: A service defines a set of pods and a means by which to access them, such as single stable IP address and corresponding DNS name.

[**Volume**](volumes.md)
: A volume is a directory, possibly with some data in it, which is accessible to a Container as part of its filesystem.  Kubernetes volumes build upon [Docker Volumes](https://docs.docker.com/userguide/dockervolumes/), adding provisioning of the volume directory and/or device.

[**Secret**](secrets.md)
: A secret stores sensitive data, such as authentication tokens, which can be made available to containers upon request.

[**Name**](identifiers.md)
: A user- or client-provided name for a resource.

[**Namespace**](namespaces.md)
: A namespace is like a prefix to the name of a resource. Namespaces help different projects, teams, or customers to share a cluster, such as by preventing name collisions between unrelated teams.

[**Annotation**](annotations.md)
: A key/value pair that can hold larger (compared to a label), and possibly not human-readable, data, intended to store non-identifying auxiliary data, especially data manipulated by tools and system extensions.  Efficient filtering by annotation values is not supported.

## Further reading

* API resources
  * [Working with resources](working-with-resources.md)

* Pods and containers
  * [Pod lifecycle and restart policies](pod-states.md)
  * [Lifecycle hooks](container-environment.md)
  * [Compute resources, such as cpu and memory](compute-resources.md)
  * [Specifying commands and requesting capabilities](containers.md)
  * [Downward API: accessing system configuration from a pod](downward-api.md)
  * [Images and registries](images.md)
  * [Migrating from docker-cli to kubectl](docker-cli-to-kubectl.md)
  * [Tips and tricks when working with config](config-best-practices.md)
  * [Assign pods to selected nodes](node-selection/)
  * [Perform a rolling update on a running group of pods](update-demo/)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
