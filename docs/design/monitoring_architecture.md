# Kubernetes monitoring architecture

## Executive Summary

Monitoring is split into two pipelines:

* A **core metrics pipeline** consisting of Kubelet, a resource estimator, a slimmed-down
Heapster called metrics-server, and the API server serving the master metrics API. These
metrics are used by core system components, such as scheduling logic (e.g. scheduler and
horizontal pod autoscaling based on system metrics) and simple out-of-the-box UI components
(e.g. `kubectl top`). This pipeline is not intended for integration with third-party
monitoring systems.
* A **monitoring pipeline** used for collecting various metrics from the system and exposing
them to end-users, as well as to the Horizontal Pod Autoscaler (for custom metrics) and Infrastore
via adapters. Users can choose from many monitoring system vendors, or run none at all. In
open-source, Kubernetes will not ship with a monitoring pipeline, but third-party options
will be easy to install. We expect that such pipelines will typically consist of a per-node
agent and a cluster-level aggregator.

The architecture is illustrated in the diagram in the Appendix of this doc.

## Introduction and Objectives

This document proposes a high-level monitoring architecture for Kubernetes. It covers
a subset of the issues mentioned in the “Kubernetes Monitoring Architecture” doc,
specifically focusing on an architecture (components and their interactions) that
hopefully meets the numerous requirements. We do not specify any particular timeframe
for implementing this architecture, nor any particular roadmap for getting there.

### Terminology

There are two types of metrics, system metrics and service metrics. System metrics are
generic metrics that are generally available from every entity that is monitored (e.g.
usage of CPU and memory by container and node). Service metrics are explicitly defined
in application code and exported (e.g. number of 500s served by the API server). Both
system metrics and service metrics can originate from users’ containers or from system
infrastructure components (master components like the API server, addon pods running on
the master, and addon pods running on user nodes).

We divide system metrics into

* *core metrics*, which are metrics that Kubernetes understands and uses for operation
of its internal components and core utilities -- for example, metrics used for scheduling
(including the inputs to the algorithms for resource estimation, initial resources/vertical
autoscaling, cluster autoscaling, and horizontal pod autoscaling excluding custom metrics),
the kube dashboard, and “kubectl top.” As of now this would consist of cpu cumulative usage,
memory instantaneous usage, disk usage of pods, disk usage of containers
* *non-core metrics*, which are not interpreted by Kubernetes; we generally assume they
include the core metrics (though not necessarily in a format Kubernetes understands) plus
additional metrics.

Service metrics can be divided into those produced by Kubernetes infrastructure components
(and thus useful for operation of the Kubernetes cluster) and those produced by user applications.
Service metrics used as input to horizontal pod autoscaling are sometimes called custom metrics.
Of course horizontal pod autoscaling also uses core metrics.

We consider logging to be separate from monitoring, so logging is outside the scope of
this doc.

### Requirements

The monitoring architecture should

* include a solution that is part of core Kubernetes and
  * makes core system metrics about nodes, pods, and containers available via a standard
  master API (today the master metrics API), such that core Kubernetes features do not
  depend on non-core components
  * requires Kubelet to only export a limited set of metrics, namely those required for
  core Kubernetes components to correctly operate (this is related to #18770)
  * can scale up to at least 5000 nodes
  * is small enough that we can require that all of its components be running in all deployment
  configurations
* include an out-of-the-box solution that can serve historical data, e.g. to support Initial
Resources and vertical pod autoscaling as well as cluster analytics queries, that depends
only on core Kubernetes
* allow for third-party monitoring solutions that are not part of core Kubernetes and can
be integrated with components like Horizontal Pod Autoscaler that require service metrics

## Architecture

We divide our description of the long-term architecture plan into the core metrics pipeline
and the monitoring pipeline. For each, it is necessary to think about how to deal with each
type of metric (core metrics, non-core metrics, and service metrics) from both the master
and minions.

### Core metrics pipeline

The core metrics pipeline collects a set of core system metrics. There are two sources for
these metrics

* Kubelet, providing per-node/pod/container usage information (the current cAdvisor that
is part of Kubelet will be slimmed down to provide only core system metrics)
* a resource estimator that runs as a DaemonSet and turns raw usage values scraped from
Kubelet into resource estimates (values used by scheduler for a more advanced usage-based
scheduler)

These sources are scraped by a component we call *metrics-server* which is like a slimmed-down
version of today's Heapster. metrics-server stores locally only latest values and has no sinks.
metrics-server exposes the master metrics API. (The configuration described here is similar
to the current Heapster in “standalone” mode.)
[Discovery summarizer](../../docs/proposals/federated-api-servers.md)
makes the master metrics API available to external clients such that from the client’s perspective
it looks the same as talking to the API server.

Core (system) metrics are handled as described above in all deployment environments. The only
easily replaceable part is resource estimator, which could be replaced by power users. In
theory, metric-server itself can also be substituted, but it’d be similar to substituting
apiserver itself or controller-manager - possible, but not recommended and not supported.

Eventually the core metrics pipeline might also collect metrics from Kubelet and Docker daemon
themselves (e.g. CPU usage of Kubelet), even though they do not run in containers.

The core metrics pipeline is intentionally small and not designed for third-party integrations.
“Full-fledged” monitoring is left to third-party systems, which provide the monitoring pipeline
(see next section) and can run on Kubernetes without having to make changes to upstream components.
In this way we can remove the burden we have today that comes with maintaining Heapster as the
integration point for every possible metrics source, sink, and feature.

#### Infrastore

We will build an open-source Infrastore component (most likely reusing existing technologies)
for serving historical queries over core system metrics and events, which it will fetch from
the master APIs. Infrastore will expose one or more APIs (possibly just SQL-like queries --
this is TBD) to handle the following use cases

* initial resources
* vertical autoscaling
* oldtimer API
* decision-support queries for debugging, capacity planning,  etc.
* usage graphs in the [Kubernetes Dashboard](https://github.com/kubernetes/dashboard)

In addition, it may collect monitoring metrics and service metrics (at least from Kubernetes
infrastructure containers), described in the upcoming sections.

### Monitoring pipeline

One of the goals of building a dedicated metrics pipeline for core metrics, as described in the
previous section, is to allow for a separate monitoring pipeline that can be very flexible
because core Kubernetes components do not need to rely on it. By default we will not provide
one, but we will provide an easy way to install one (using a single command, most likely using
Helm). We described the monitoring pipeline in this section.

Data collected by the monitoring pipeline may contain any sub- or superset of the following groups
of metrics:

* core system metrics
* non-core system metrics
* service metrics from user application containers
* service metrics from Kubernetes infrastructure containers; these metrics are exposed using
Prometheus instrumentation

It is up to the monitoring solution to decide which of these are collected.

In order to enable horizontal pod autoscaling based on custom metrics, the provider of the
monitoring pipeline would also have to create a stateless API adapter that pulls the custom
metrics from the monitoring pipeline and exposes them to the Horizontal Pod Autoscaler. Such
API will be a well defined, versioned API similar to regular APIs. Details of how it will be
exposed or discovered will be covered in a detailed design doc for this component.

The same approach applies if it is desired to make monitoring pipeline metrics available in
Infrastore. These adapters could be standalone components, libraries, or part of the monitoring
solution itself.

There are many possible combinations of node and cluster-level agents that could comprise a
monitoring pipeline, including
cAdvisor + Heapster + InfluxDB (or any other sink)
* cAdvisor + collectd + Heapster
* cAdvisor + Prometheus
* snapd + Heapster
* snapd + SNAP cluster-level agent
* Sysdig

As an example we’ll describe a potential integration with cAdvisor + Prometheus.

Prometheus has the following metric sources on a node:
* core and non-core system metrics from cAdvisor
* service metrics exposed by containers via HTTP handler in Prometheus format
* [optional] metrics about node itself from Node Exporter (a Prometheus component)

All of them are polled by the Prometheus cluster-level agent. We can use the Prometheus
cluster-level agent as a source for horizontal pod autoscaling custom metrics by using a
standalone API adapter that proxies/translates between the Prometheus Query Language endpoint
on the Prometheus cluster-level agent and an HPA-specific API. Likewise an adapter can be
used to make the metrics from the monitoring pipeline available in Infrastore. Neither
adapter is necessary if the user does not need the corresponding feature.

The command that installs cAdvisor+Prometheus should also automatically set up collection
of the metrics from infrastructure containers. This is possible because the names of the
infrastructure containers and metrics of interest are part of the Kubernetes control plane
configuration itself, and because the infrastructure containers export their metrics in
Prometheus format.

## Appendix: Architecture diagram

### Open-source monitoring pipeline

![Architecture Diagram](monitoring_architecture.png?raw=true "Architecture overview")



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/monitoring_architecture.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
