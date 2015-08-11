---
layout: docwithnav
title: "Kubernetes User Guide: Managing Applications"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


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

The user guide is intended for anyone who wants to run programs and services on an existing Kubernetes cluster.  Setup and administration of a Kubernetes cluster is described in the [Cluster Admin Guide](../../docs/admin/README.html). The [Developer Guide](../../docs/devel/README.html) is for anyone wanting to either write code which directly accesses the Kubernetes API, or to contribute directly to the Kubernetes project.

Please ensure you have completed the [prerequisites for running examples from the user guide](prereqs.html).

## Quick walkthrough

1. [Kubernetes 101](walkthrough/README.html)
1. [Kubernetes 201](walkthrough/k8s201.html)

## Thorough walkthrough

If you don't have much familiarity with Kubernetes, we recommend you read the following sections in order:

1. [Quick start: launch and expose an application](quick-start.html)
1. [Configuring and launching containers: configuring common container parameters](configuring-containers.html)
1. [Deploying continuously running applications](deploying-applications.html)
1. [Connecting applications: exposing applications to clients and users](connecting-applications.html)
1. [Working with containers in production](production-pods.html)
1. [Managing deployments](managing-deployments.html)
1. [Application introspection and debugging](introspection-and-debugging.html)
    1. [Using the Kubernetes web user interface](ui.html)
    1. [Logging](logging.html)
    1. [Monitoring](monitoring.html)
    1. [Getting into containers via `exec`](getting-into-containers.html)
    1. [Connecting to containers via proxies](connecting-to-applications-proxy.html)
    1. [Connecting to containers via port forwarding](connecting-to-applications-port-forward.html)

## Concept guide

[**Overview**](overview.html)
: A brief overview of Kubernetes concepts.

[**Cluster**](../admin/README.html)
: A cluster is a set of physical or virtual machines and other infrastructure resources used by Kubernetes to run your applications.

[**Node**](../admin/node.html)
: A node is a physical or virtual machine running Kubernetes, onto which pods can be scheduled.

[**Pod**](pods.html)
: A pod is a co-located group of containers and volumes.

[**Label**](labels.html)
: A label is a key/value pair that is attached to a resource, such as a pod, to convey a user-defined identifying attribute. Labels can be used to organize and to select subsets of resources.

[**Selector**](labels.html#label-selectors)
: A selector is an expression that matches labels in order to identify related resources, such as which pods are targeted by a load-balanced service.

[**Replication Controller**](replication-controller.html)
: A replication controller ensures that a specified number of pod replicas are running at any one time. It both allows for easy scaling of replicated systems and handles re-creation of a pod when the machine it is on reboots or otherwise fails.

[**Service**](services.html)
: A service defines a set of pods and a means by which to access them, such as single stable IP address and corresponding DNS name.

[**Volume**](volumes.html)
: A volume is a directory, possibly with some data in it, which is accessible to a Container as part of its filesystem.  Kubernetes volumes build upon [Docker Volumes](https://docs.docker.com/userguide/dockervolumes/), adding provisioning of the volume directory and/or device.

[**Secret**](secrets.html)
: A secret stores sensitive data, such as authentication tokens, which can be made available to containers upon request.

[**Name**](identifiers.html)
: A user- or client-provided name for a resource.

[**Namespace**](namespaces.html)
: A namespace is like a prefix to the name of a resource. Namespaces help different projects, teams, or customers to share a cluster, such as by preventing name collisions between unrelated teams.

[**Annotation**](annotations.html)
: A key/value pair that can hold larger (compared to a label), and possibly not human-readable, data, intended to store non-identifying auxiliary data, especially data manipulated by tools and system extensions.  Efficient filtering by annotation values is not supported.

## Further reading

* API resources
  * [Working with resources](working-with-resources.html)

* Pods and containers
  * [Pod lifecycle and restart policies](pod-states.html)
  * [Lifecycle hooks](container-environment.html)
  * [Compute resources, such as cpu and memory](compute-resources.html)
  * [Specifying commands and requesting capabilities](containers.html)
  * [Downward API: accessing system configuration from a pod](downward-api.html)
  * [Images and registries](images.html)
  * [Migrating from docker-cli to kubectl](docker-cli-to-kubectl.html)
  * [Tips and tricks when working with config](config-best-practices.html)
  * [Assign pods to selected nodes](node-selection/)
  * [Perform a rolling update on a running group of pods](update-demo/)


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

