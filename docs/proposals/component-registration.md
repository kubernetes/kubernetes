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
[here](http://releases.k8s.io/release-1.0/docs/proposals/component-registration.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Proposal: Component Registration

## Table of Contents

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Proposal: Component Registration](#proposal-component-registration)
  - [Table of Contents](#table-of-contents)
  - [Terminology](#terminology)
  - [Abstract](#abstract)
  - [Problems](#problems)
    - [ComponentStatuses](#componentstatuses)
    - [Endpoints](#endpoints)
  - [Users](#users)
  - [Use Cases](#use-cases)
    - [Deployment](#deployment)
    - [Debugging](#debugging)
    - [Discovery](#discovery)
    - [Notification](#notification)
    - [Reporting](#reporting)
    - [Self-Healing](#self-healing)
  - [Design](#design)
    - [Phased Changes](#phased-changes)
    - [API Changes](#api-changes)
      - [API Types](#api-types)
    - [Kubectl (CLI)](#kubectl-cli)
    - [Storage](#storage)
    - [Status Updates](#status-updates)
  - [Example](#example)
  - [Alternate Naming](#alternate-naming)
  - [Next Steps](#next-steps)

<!-- END MUNGE: GENERATED_TOC -->

## Terminology

A Kubernetes **Component**...

- Must know about Kubernetes (speak the API)
- Must register (or be registered) with Kubernetes (apiserver)
- Must be deployed/managed by an admin/operator (rather than a user)
- Must expose a method for remotely probing the liveness and readiness of each instance (e.g. tcp/http/https).
- May be scheduled/deployed by Kubernetes
  - Either on the cluster with a ReplicationControler
  - Or on a single node/kubelet as a Daemon
- May be scheduled/deployed by an external system (not Kubernetes)
- May be required for the cluster to schedule Pods (or may add additional functionality)
- May be registered/unregistered at runtime (mutable cluster component membership over time)
- May require admin API access (privileged behavior)
- May consist of one or more instances
- May be deployed on different (network accessible) machines
- May be upgraded over time

The **External Components** are components that are deployed and lifecycle managed by some external system.

The **Addon Components** are components that are deployed and lifecycle managed by Kubernetes to which they provide functionality. These components must be optional in order for them to be deployable by Kubernetes (at least until an HA bootstrapping method is developed). Kubernetes conformance tests currently require some of these (e.g. kube-dns), but they are not required for orchestration and scheduling functionality.

The **Primary Components** (currently apiserver, controller-manager, and scheduler) are components that are required for orchestration and scheduling functionality. These components are currently all External Components but may become Addon Components, once an HA bootstrapping method is developed.

The"**Hyperkube** is a single binary that can be run as each of the three primary component binaries. Hyperkube may also refer to the set (or pod) of primary component containers or the (virtual) machine that those containers run on. Not all Kubernetes deployments use or should be required to use the Hyperkube.

**Master** has been sometimes used in reference to both the APIServer component and the (virtual) machine which hosts the three primary components. I'm going to avoid this term, because it is ambiguous.

**Etcd** is a consistent key value storage cluster. It should not be considered a component, because it is a dependency of the apiserver, does not know about Kubernetes, and thus cannot register itself with Kubernetes but must be configured. In general, the apiserver attempts to hide/abstract Etcd from the rest of the components with its API.

**ComponentStatuses** is a top-level Kubernetes API resource that exposes a `/componentstatuses` endpoint that shows component and Etcd status. It's unquestionably useful and used, but questionably named and implemented. However, it must continue to function (until v2) for reverse compatibility. See [Problems](#problems) for more details. `kubectl get componentstatuses` can be used from the CLI as a wrapper for the API..

**Liveness** is a condition of a service instance that indicates it is running. When a local process is NotAlive it needs to be restarted. When a remote service is NotAlive it may been to be restarted, or the networking may need to be fixed.

**Readiness** is a condition of a service instance that indicates it is Alive, able to handle new requests, and that its dependencies are also Ready. Ideally transitivity of readiness requires that only immediate dependencies be confirmed to be Ready. Readiness is usually expected to be easily assessed, slightly more expensive than liveness assessment, but still cheap enough to perform frequently and continuously. Readiness is not expected to provide robust metrics data, which may be more expensive to gather.

**Canary Conditions** are conditions which indicate problems. Liveness and Readiness are canary conditions because when failing, they indicate problems, but do not guarantee health when not failing. Because of this distinction, and to avoid ambiguity, the term "health" should be avoided in this context.

**Endpoints** is a namespaced Kubernetes API that describes a set of address:port pairs (with optional port name) that describe the providing instances of a Service. Currently only Ready addresses are included, but there is a PR to add NotReady addresses: #13778. Internal Endpoints are created and managed automatically by the Endpoints Controller. External Endpoints are created and managed manually by Kubernetes users.

An **Internal Service** is a Service defined by Pod Selectors. Internal Endpoints for each Internal Service are created and managed automatically by the Endpoints Controller based on ServiceSpec Selectors.

An **External Service** is a Service without Selectors. External Endpoints for each External Service are created and managed manually (or at least are not currently created/managed by primary Kubernetes components).


## Abstract

The goal of this proposal is to move Kubernetes a little closer to each of the following goals:

- **High Availability**
  - Allow multiple instances of each component
- **Extensibility**
  - Allow optional and third party components to be added to the cluster
- **Deployment Flexibility**
  - Allow external components to be deployed on different machines (or even different availability zones)
- **Operability**
  - Expose a filterable list of component conditions on the apiserver to simplify collection of component conditions
  - Expose a filterable list of components on the apiserver to see which components are installed/deployed
- **Security**
  - Cache component conditions in Etcd (via apiserver) to reduce denial of service risk caused by network fan-out (requires continuous liveness/readiness probing)
- **Consistency/Simplicity/Reuse**
  - Make external services more like internal services (defined by selectors of another resource)
  - Simplify the usage of the Endpoints API to be exclusively dynamically generated (read-only)
  - Group the Specification (Spec) and Status of external service provider instances into a single model (Endpoints is messy combination of both Spec and Status for both internal and external addresses)
  - Register components as external services (namespaced and labeled)
  - Implement liveness/readiness probing in a way that can be used by all external services, not just components


## Problems

### ComponentStatuses

1. The set of components is hardcoded
2. The component addresses and ports are hardcoded (localhost, default port)
3. The `/healthz` probe path is hardcoded
4. Etcd is included as a component
5. Each user request performs network amplification by fanning out requests to all components (DOS risk)
6. Probing is done in serial (slow)
7. Inconsistent with other registry/storage APIs and their implementations


### Endpoints

1. Endpoints doesn't distinguish between Spec and Status. In most other resources the Spec is the desired state and the Status is current (or last known) state. For more details, see the [API Conventions](../devel/api-conventions.md#spec-and-status).
2. Endpoints is both a dynamically populated resource AND a manually populated resource. This confusion of ownership makes it hard to use, hard to describe, overly complicated, and inconsistent with other API resources.

## Users

Here we'll define several Kubernetes user types for the purposes of clarifying Use Cases.

- **Kubernetes Operator** - one who manages/administers a running Kubernetes cluster (including deployment)
- **Kubernetes Client** - one who queries the Kubernetes API directly or with kubectl (without requiring admin privileges)

## Use Cases

This proposal primarily covers features that can be achieved internally by Kubernetes itself, but many of these features and changes are motivated by external use cases, which need enablement. Not all of these use cases will be completely enabled by this proposal.


### Deployment

As a Kubernetes Operator, I want a to be able to add/remove new components (instances or types) to a new or running cluster (ideally without manually registering it with the apiserver).

As a Kubernetes Operator, I want a kubectl command that allows me to validate that the cluster has been deployed and is fully functional.

As a Kubernetes Operator, I want a kubectl command that allows me to check which components are deployed/installed on a specific cluster.

### Debugging

As a Kubernetes Operator, I want a kubectl command that shows me the current condition of the cluster's components, so that I can debug observed failures.

### Discovery

As a Kubernetes Client, I want kubectl command that lists (optionally namespaced) services/components that provide specific functionality (e.g. DNS, logging, metrics, or cluster backup).

### Notification

As a Kubernetes Operator, I want to be notified when the cluster requires human intervention, so that I can intervene.

As a service that provides notifications (e.g. PagerDuty), I want an to retrieve the current condition of the cluster's components, so that I can determine (with my own rule set) when to notify the Kubernetes Operator.

### Reporting

As a Kubernetes Operator, I want to see a report/graph of the readiness/uptime/availability of the cluster over time, so that I can make/satisfy SLAs.

As a service that provides reports/graphs (e.g. DataDog), I want to retrieve status metrics for each component, so that I can record and report on the metrics over time.

### Self-Healing

As a Kubernetes Operator, I want the cluster to self-heal when individual components crash so that I don't get notified or need to intervene.

As a process monitoring tool (e.g. monit), I want to be able to determine the condition of the component process I am monitoring to determine if it needs to be restarted.

As a container/vm monitoring tool (e.g. Marathon, Amazon CloudWatch), I want to be able to determine the health of the component in the container/vm I am monitoring.

Note: Self-healing external components requires knowing how to deploy the external components, which is different for each cloud platform and changes based on component configuration. This is different than self-healing addon components, which is covered by the Kubernetes replication controller.


## Design

Unfortunately, ComponentStatuses is almost not worth the effort to fix. Instead it can be deprecated and replaced by a combination of Service and Endpoints changes. This allows the components and their addresses & ports to be specified more flexibly.

Since it's seemingly impossible to fix Endpoints Problem \#1 without breaking reverse compatibility, this proposal punts on \#1 (until v2 allows reverse-incompatibility) and focuses on fixing \#2 for now. #13778 handles increasing the scope of Endpoints to cover both Ready and NotReady addresses, which is better than nothing, but not as flexible as having an array of conditions, like other -Status API resources.

To fix Endpoints Problem \#2, this proposal moves the specification (and status) of external service provider addresses to its own namespaced API resource (provisionally called **ExternalServiceProvider**). Endpoints Controller will then continue to manage the lifecycle (create/update/delete) of Internal Endpoints based on a selection of Pod instances, but will also do the same for External Endpoints based on a selection of ExternalServiceProviders defined in ServiceSpec.Selector. Endpoints then becomes a (conceptually) read-only API for the user. Endpoints can't actually be read-only, because the Endpoints Controller needs access, but write access may eventually be limited to admin/operator-only.


### Phased Changes

Since the changes here are complex, they will be implemented in phases.

1. Simplify Endpoints
  - Add a new API resource (ExternalServiceProvider) that represents a non-pod service provider
    - Specify a network route (host[:port])
    - Specify liveness and readiness probe methods
    - Store the last known status (list of conditions)
  - Add (or modify) a controller to probe ExternalServiceProviders and update their status
  - Update EndpointsController to manage Endpoints based on both Pod and ExternalServiceProvider status
2. Register component ExternalServiceProviders
  - Update all components to use consistent namespacing and component labeling
  - EITHER update "external" components to self-register their ExternalServiceProvider
  - OR update deployment automation to register deployed components as ExternalServiceProviders
3. Add Service Status Conditions
  - Add Service condition policies (all, half+1, at least n) that can be specified for each Service
  - Add (or modify) a controller to update the Service status conditions based on aggregate provider (Pod/ExternalServiceProvider) status
  - Make sure Services can be filtered by status conditions (e.g. all ready services with label x)
  - Deprecate ComponentStatuses and migrate users to filtered queries (`type=component`)
    - For aggregate status query ServiceList
    - For individual service provider status query PodList & ExternalServiceProviderList
4. Optimize querying service provider spec + status (PodList + ExternalServiceProviderList)
  - Either add a new on-demand "joined view" meta-resource
  - OR add a new Component resource (read-only cache) that gets updated every time `type=component` Pods or ExternalServiceProviders get updated
5. Expand Service or add new a new API resource to allow discovery of API group providers


### API Changes

1. Add new namespaced API resource: ExternalServiceProvider
  - See [Alternate Naming](#alternate-naming)
  - Similar to Pods
    - Listable, like most other top-level API resources
    - Have both Spec and Status (filterable on both/either)
    - Spec would have (optional) LivenessProbe and ReadinessProbe for determining the condition
    - Describes a single instance
    - Can be selected by labels, metadata, spec, and status
    - Is namespaced (components would use a specific namespace, likely "kube-system")
    - Status would have an array of [Conditions](../devel/api-conventions.md#typical-status-properties)
    - Can be named with a name generator (similar to pods created by replication controllers) when instances of the same type don't already have unique IDs (generated name is returned in the create response)
  - Similar to Endpoints (current External Endpoints usage)
    - Spec would have address (ip or host) and ports (optionally named) to connect with an external service provider instance
    - Manually added for non-component External Service provider instances (e.g. Oracle DB) that do not know about k8s (rather than adding Endpoints)
  - Different from Endpoints
    - Singular, instead of plural (use -List, namespacing, and labels for grouping)
    - Includes both Ready & NotReady endpoints
    - Supports multiple and future conditions (in an extensible Conditions array)
2. Modify ServiceSpec.Selector to select from both Pods and ExternalServiceProviders
3. Add a new ExternalServiceProvider Controller that handles probing ExternalServiceProviders
  - Probe results update ExternalServiceProvider.Status.Conditions
4. Modify Endpoints Controller to manage ExternalServiceProvider Endpoints
  - Readiness from ExternalServiceProvider.Status.Conditions analogous to PodStatus.Conditions
5. Modify the Primary Components to self-register ExternalServiceProviders (one per component instance) instead of an Endpoint
  - Group ExternalServiceProviders (component instances) with labels (e.g. "type=component", "component-type=core")
6. Modify the Addon Components' Pod and Service registration to use similar labels (e.g. "type=component", "component-type=addon")
7. Register component Services (with ServiceSpec.Selector) as part of cluster deployment
  - Services of new components can also be added at runtime, before any instances exist. Each instance would self-register a ExternalServiceProviders.
8. Deprecate `/componentsstatuses` in favor of requesting a filtered list of ExternalServiceProviders
  - ExternalServiceProviders are external-only, but have generic conditions (liveness and readiness)
  - Endpoints are external + internal, but only have readiness (for now)


#### API Types

The following is a draft of how the API models might change.

```
type ExternalServiceProvider struct {
	TypeMeta
	ObjectMeta
	Spec ExternalServiceProviderSpec
	Status ExternalServiceProviderStatus
}

type ExternalServiceProviderSpec struct {
	Host string
	Ports []ExternalServiceProviderPort
	LivenessProbe *Probe
	ReadinessProbe *Probe
}

type ExternalServiceProviderPort struct {
	Name string // optional - for mapping from ServicePort
	Protocol Protocol // TCP, UDP
	Port int
}

type ExternalServiceProviderStatus struct {
	Conditions []ExternalServiceProviderCondition
}

type ExternalServiceProviderCondition struct {
	Type ExternalServiceProviderConditionType // Ready, Alive
	Status ConditionStatus // True, False, Unknown
	LastUpdateTime util.Time
	LastTransitionTime util.Time
	Reason string
	Message string
}

type ExternalServiceProviderList struct {
	TypeMeta
	ListMeta
	Items []ExternalServiceProvider
}
```

```
type ServiceSpec struct {
	...
	ConditionPolicies []ServiceConditionPolicy
}

type ServiceConditionPolicy struct {
	Type ServiceConditionType
	MinNumber *int
	MinPercent *float
}

type ServiceStatus struct {
	...
	Conditions []ServiceCondition
}

type ServiceCondition struct {
	Type ServiceConditionType // Ready, Alive
	Status ConditionStatus // True, False, Unknown
	LastUpdateTime util.Time
	LastTransitionTime util.Time
	Reason string
	Message string
}
```


### Kubectl (CLI)

Ready Service Endpoints can be queried with the existing `kubectl get endpoints <service-name>`.

Add a `kubectl get externalserviceproviders` (`kubectl get esp` for short) command to list the external service provider instances, their address/port/name and their last known conditions. Components can be retrived with a component specific label, like `type=component` or `component=apiserver`.


### Storage

Store ExternalServiceProviders spec/status in etcd, with an apiserver storage wrapper.


### Status Updates

Add an external-service-provider-controller that regularly probes registered providers and updates their stored last known status.

Add self-registration to the components that uses the new ExternalServiceProviders API (Create call) when the component starts.

Add self-deregistration to the components that uses the new ExternalServiceProviders API (Remove call) when the component stops gracefully (e.g. after receiving SIGTERM).

Add self-updating to the components that uses the new ExternalServiceProviders API (Update call) when the component errors/panics.

Use a TCP probe (of `/healthz` or `/healthz/ping`) to check liveness, because it indicates a server is listening on the expected host/path, even if its responses are unhealthy.

Update individual component `/healthz` endpoints with sub-resources to allow them to perform as an indicator of readiness, not just liveness. Readiness should include transitively checking the readiness of immediate dependencies (e.g. etcd or other databases).


## Example

The following terminal commands are an example of how the `kubectl get externalserviceproviders` command might work.

TODO

## Alternate Naming

ExternalServiceProvider is a bit verbose for a top-level API resource name. Here are some other naming options for the same concept.

ExternalTarget
[External]Endpoint (singular)
[External]ServiceTarget
[External]ServiceInstance
[External]ServiceTarget
[External]Shell (Pod without peas)
[External]ServiceIngress
[External]ServiceUnit
[External]ServiceMember

We considered ExternalPod, but figured it would be too confusing, because it doesn't have to be implemented by a Pod.

Including Endpoint in the name is also pretty confusing unless we deprecate and replace Endpoints over time.


## Next Steps

Since existing AddOns (e.g. kube-ui, kube-dns) already register themselves as internal services, this proposal puts them on a similar footing to externally hosted components.

Services could be made to include both internal pods and external providers (in the same service).

Because this proposal includes API changes, it might be put in either a new API group or in a new API version. However, the intent was to make changes that were backwards compatible, at least for the beginning phases.

Because of the desire for reverse compatibility, the existing `/componentstatuses` endpoint will likely be deprecated and eventually removed. We *could* re-implement it using the API added by this proposal, but it's more desirable to also update the API to match existing conventions, which would be effectively equivalent to deleting it and making a new API (e.g. `/components/status`).

Another interesting option is to add a generic or specific join functionality (think relational DB) that could join Service + ExternalServiceProvider + Pod for filterable queries, without having specific storage for the result. This would make the Endpoints API obsolete.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/component-registration.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
