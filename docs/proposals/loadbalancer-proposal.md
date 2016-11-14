<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.4/docs/proposals/federation.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Ingress, LoadBalancer and LoadBalancerClaim proposal

@mqliang, @ddysher

Nov 2016

## Overview

The proposal aims to define a set of APIs to abstract loadbalancer resource,
similar to how PV/PVC works. In the end, we'll

* introduce a new API object `LoadBalancer` to represent a loadbalancer (think of PV)
* introduce a new API object `LoadBalancerClaim` to claim a new or existing
  LoadBalancer (think of PVC)
* change existing Ingress API to include a `LBSource` field, which either directly
  uses a LoadBalancer, or via a LoadBalancerClaim (think of Pod using PV or PVC)

We will also introduce network resource such as bandwidth and iops. LoadBalancerClaim
uses these attributes to claim a LoadBalacer. Though similar to PV/PVC, it is
important to know that LoadBalancer and LoadBalancerClaim is not 1-to-1 binding;
rather, it is a 1-to-many relationship. That is, a LoadBalancer can be used to
serve multiple LoadBalancerClaim.

## Background

### Current Ingress behavior

Ingress can be used to expose a service in the kubernetes cluster:

* cluster admin deploys an ingress-controller Pod beforehand
* user creates Ingress resource
* the ingress-controller Pod list&watch **All** Ingress Resources in the cluster,
  when it sees a new Ingress resource:
  * on cloud provider, it calls the cloud provider to sync the ingress L7
    loadbalancing rules
  * on bare-metal, it syncs nginx (or haproxy, etc) config then reload
* user out of cluster can then access service in the cluster:
  * on bare-metal, accessing the node's ip on which ingress-controller Pod is
    running, ingress-controller Pod will forward request into cluster based on
    rules defined in Ingress resource
  * on cloud-provider, accessing the ip provided by cloud provider loadbalancer,
    cloud provider will forward request into cluster based on rules defined in
    Ingress Resource

### Limitations of current Ingress implementation

* How many ingress-controller Pods should run in a cluster? Should all
  ingress-controller Pod list&watch all Ingress resource? There is no way to
  bind or schedule Ingress resource to a ingress-controller Pod, which result in:
   * insufficient or excessive use of neworking resource
   * reload storm when updating Ingress resource, or due to Pod changes
* Ingress resource is actually internet l7 loadbalancing rules, intranet l7
  loadbalancing rules has not been supported yet. Eventually, We need a general
  mechanism for both the Ingress and intranet L7 lb rules "consume" a
  loadbalancer
* On bare-metal, it does not provide High Availability because client needs
  to know the IP addresss of the node where ingress-controller Pod is running.
  In case of a failure, the ingress-controller Pod will be moved to a different
  node.

## Goal and NonGoal

### Goal

* Define `LoadBalancer` API
* Define `LoadBalancerClaim` API
* Define network attributes of LoadBalancer
* Define loadbalancer provider, what it is and how it works
* Define loadbalancer scheduling

### NonGoal

* LoadBalancer HA on bare-metal
* LoadBalancerClass: different types (internet or intranet), different qos level, etc
* LoadBalancer scheduling and over-commitment

## Design

### LoadBalancer

`LoadBalancer` is a first-class API in kubernetes. It represents network resource
for internet and intranet loadbalance. LoadBalancer will eventually be 'used' or
'consumed' via Ingress resources, which basically defines forwarding rules to a set
of Pods. Different LoadBalancer has different network attributes (bandwidth, iops,
etc).

### LoadBalancerClaim

`LoadBalancerClaim` is also a first-class API in kubernetes, and as its name
suggests, it is used to claim a LoadBalancer. LoadBalancerClaim claims a LoadBalancer
based on network attributes mentioned above. If no LoadBalancer satisfies a
claim, a new one can be created on the fly, just like how PV is dynamically
provisioned.

For more background see https://github.com/kubernetes/kubernetes/issues/30151

### LoadBalancer Provider Interface

LoadBalancer Provider Interface is an interface to Create/Update/Delete LoadBalancer.
There can be multiple LoadBalancer provider implementations, such as:
* AWS loadbalancer provider
* GCE loadbalancer provider
* bare-metal nginx loadbalancer provider
* bare-metal highly-available nginx loadbalancer provider
* bare-metal haproxy loadbalancer provider
* bare-metal highly-available haproxy loadbalancer provider

### Loadbalancer-controller

loadbalancer-controller is responsible for:

* list&watch Ingress resources and call Loadbalancer provider to update
  corresponding loadbalancer's loadbalancing rules
* pick a best-matching LoadBalancer from existing LoadBalancer pool for
  LoadBalancerClaim based on network attribute request of the LoadBalancerClaim
* call loadbalancer provider to dynamically provision a LoadBalancer for
  LoadBalancercLaim when it can not find a matching one among existing LoadBalancer
* recycle or deprovision a LoadBalancer when there is no consumers

### Loadbalancer scheduling

As mentioned before, LoadBalancer and LoadBalancerClaim binding is not exclusive,
which means multiple LoadBalancerClaim can be bound to one LoadBalancer. For
example, if we have a LoadBalancer with 3G bandwidth, we can bind 6 LoadBalancerClaim
each request 500m bandwidth on it. In such case, we need to a 'scheduling' logic.

Further, we can eventually introduce the 'request/limit model' for network resource
to acheieve functionalities already implemented in compute resource, for example,
qos and overcommit.

##### Manually assign LoadBalancerClaim to a LodBalancer
User can also manually assign LoadBalancerClaim to a LodBalancer, instead of letting
loadbalancer-controller schedule for him. In such a case, resource request of all
loadbalancerclaims may excess the loadbalancer's capacity. We validate request and
capacity when loadbalancer-controller updating the loadbalancing rules: sort all
Ingress "consume" a same LoadBalancers by creation time, if the sum request excess
loadbalancer's capacity, avoid updating rules for the last few Ingress and send a
event. Just like how we validate request at kubelet's side for Pods.


## API

### Network Resource

```go
const (
	ResourceBandWidth ResourceName = "network-bandwidth"
	ResourceIOPS      ResourceName = "network-iops"
)
```

We can introduce more network resource in the future.

### Loadbalancer API

```go
type LoadBalancer struct {
	unversioned.TypeMeta `json:",inline"`
	ObjectMeta           `json:"metadata,omitempty"`

	// Spec defines a loadbalancer owned by the cluster
	Spec LoadBalancerSpec `json:"spec,omitempty"`

	// Status represents the current information about loadbalancer.
	Status LoadBalancerStatus `json:"status,omitempty"`
}

type LoadBalancerSpec struct {
	// Resources represents the actual resources of the loadbalancer
	Capacity ResourceList `json:"capacity"`
	// Source represents the location and type of a loadbalancer to use.
	LoadBalancerSource `json:",inline"`
}

type LoadBalancerSource struct {
	GCELoadBalancer       *GCELoadBalancerSource       `json:"gceLoadBalancer,omitempty"`
	AWSLoadBalancer       *AWSLoadBalancerSource       `json:"awsLoadBalancer,omitempty"`
	BareMetalLoadBalancer *BareMetalLoadBalancerSource `json:"bareMetalLoadBalancer,omitempty"`
	/*
		more loadbalancer source
	*/
}

type GCELoadBalancerSource struct {
	// Unique name of the LoadBalancer resource. Used to identify the LoadBalancer in GCE
	LBName string `json:"lbName"`
}

type AWSLoadBalancerSource struct {
	// Unique name of the LoadBalancer resource. Used to identify the LoadBalancer in AWS
	LBName string `json:"lbName"`
}

type BareMetalLoadBalancerSource struct {
	Type        BareMetalLoadBalancerType `json:"type"`
	ServiceName string                    `json:"serviceName"` //BareMetalLoadBalancer is actually a nginx or haproxy app deploymented in "kube-system" namespace
}

type BareMetalLoadBalancerType string

const (
	NginxLoadBalancer   BareMetalLoadBalancerType = "Nginx"
	HaproxyLoadBalancer BareMetalLoadBalancerType = "Haproxy"
)

// LoadBalancerStatus represents the status of a load-balancer
type LoadBalancerStatus struct {
	// Phase indicates if a loadbalancer is pending, running or failed
	Phase LoadBalancerPhase `json:"phase,omitempty"`
	// A human-readable message indicating details about why the loadbalancer is in this state.
	Message string `json:"message,omitempty"`
	// Ingress is a list containing ingress points for the load-balancer;
	// traffic should be sent to these ingress points.
	Ingress []LoadBalancerIngress `json:"ingress,omitempty"`
}

type LoadBalancerPhase string

const (
	// used for LoadBalancers that are not available
	LoadBalancerPending LoadBalancerPhase = "Pending"
	// used for LoadBalancers that are working well
	LoadBalancerPending LoadBalancerPhase = "Running"
	// used for LoadBalancers that failed to be correctly recycled or deleted after being released from a claim
	LoadBalanceFailed LoadBalancerPhase = "Failed"
)

// LoadBalancerIngress represents the status of a load-balancer ingress point:
// traffic should be sent to an ingress point.
type LoadBalancerIngress struct {
	// IP is set for load-balancer ingress points that are IP based
	// (typically GCE or OpenStack load-balancers)
	IP string `json:"ip,omitempty"`

	// Hostname is set for load-balancer ingress points that are DNS based
	// (typically AWS load-balancers)
	Hostname string `json:"hostname,omitempty"`
}
```

### LoadbalancerClaim API

```go
type LoadbalancerClaim struct {
	unversioned.TypeMeta `json:",inline"`
	ObjectMeta           `json:"metadata,omitempty"`

	// Spec defines a loadbalancer owned by the cluster
	Spec LoadBalancerClaimSpec `json:"spec,omitempty"`

	// Status represents the current information about loadbalancer.
	Status LoadBalancerClaimStatus `json:"status,omitempty"`
}

type LoadBalancerClaimSpec struct {
	// A label query over loadbalancer to consider for binding. This selector is
	// ignored when LoadBalancerName is set
	Selector *unversioned.LabelSelector `json:"selector,omitempty"`
	// Resources represents the minimum resources required
	Resources ResourceRequirements `json:"resources,omitempty"`
	// LoadBalancerName is the binding reference to the LoadBalancerName backing this
	// claim. When set to non-empty value Selector is not evaluated
	LoadBalancerName string `json:"loadbalancerName,omitempty"`
}

type LoadBalancerClaimStatus struct {
}
```

### Ingress API Changes

Add a new knob "LoadBalancer" in IngressSpec, so that Ingress can "use" or
"consume" a LoadBalancer, either directly or through LoadBalancerClaim.

```go
type IngressSpec struct {
    /*
    ...
    */

    LoadBalancer LBSource `json:"loadBalancer"`
}

type LBSource struct {
    GCELoadBalancer *GCELoadBalancerSource `json:"gceLoadBalancer,omitempty"`
	AWSLoadBalancer *AWSLoadBalancerSource `json:"awsLoadBalancer,omitempty"`
	BareMetalLoadBalancer *BareMetalLoadBalancerSource `json:"bareMetalLoadBalancer,omitempty"`
	LoadBalancerClaim *LoadBalancerClaimSource `json:"loadBalancerClaim,omitempty"`
}

type LoadBalancerClaimSource struct {
	// ClaimName is the name of a LoadBalancerClaim in the same namespace as the ingress using this lb
	ClaimName string `json:"claimName"`
}
```

## Implementation Detail

### loadbalancer

The current loadbalancer (ingress-controller) implementation has some limitations
when we have multiple loadbalancers:

* we need to deploy multiple ingress-controller pod even on cloud provider, it
  results in some excessive use of resource. Actually it fine that a cluster
  just has one process which list&watch all ingresses, and call cloud provider
  provider to update loadbalancing rules.

Thus, we propose to redesign the current ingress-controller as follows:

Add a loadbalancer-controller in conroller-manager component, which list&watch
all ingresses resource and call loadbalancer provider to update the
loadbalancing rules.

* On cloud provider

  No extra ingress-controler need to be deployed

* On bare-metal
    * put all loadbalancing rules in configmap
    * ingress-controller not need list&watch any Ingress anymore. Just notify
      the nginx/haproxy process to reload when configmap was updated.
    * loadbalancer-controller in controller-manager component will list&watch
      all Ingress and update corresponding nginx/haproxy's configmap

#### loadbalancer controller

Add a loadbalancer-controller in conroller-manager component. Which works as
follows:

* bind loadbalancerclaim with loadbalancer
* dynamically provision loadbalancer on demand
* recycle or deprovision a loadbalancer when no loadbalancerclaim bind with
  it for a long time.
* list&watch all ingress and call loadbalancer provider to update l7
  loadbalancing rules
* we put loadblancer provider logic in loadbalancer-controlle too. On cloud
  provider, it just delegate all loadbalancer provider logic to cloud provider;
  on bare-metal, it:
  * create a loadbalancer by deploying a nginx/haproxy pod in "kube-system"
    namespace
  * update a loadbalancer by updating the configmap of nginx/haproxy pod
  * delete a loadbalancer by deleting the nginx/haproxy pod and all its relating
    resource(such as configmap, etc).

## Implementation plan

### First step: make it workable

* implement the scheduling, provisioning and recycling logic first.
* ingress-controller just works as currently, but it just list&watch ingresses
  assigned to it, insteal all ingresses

### Second step: loadbalancer provider

* Add loadbalancer provider in loadbalancer-controller
* refactor the current ingress-controller implementation as I descrive in the
  "Implementation" section and rename "ingress-controller" as
  "nginx-loadbalancer"

### Long term

* loadbalancer scheduling over-commitment


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/federation.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
