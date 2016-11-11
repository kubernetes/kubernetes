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

# Loadbalancer and LoadbalancerClaim proposal
----

### Overview

LoadBalancer is a commen resource provided by IaaS cloud provider, but currently
kuberntes does not expose cloud LoadBalancer resource to users explicitly, we
just use it implicitly by "Loadbalancer" type Service. We proposed to:

* explicitly expose cloud loadbalancer to users.
* introduce `LoadbalancerClaim` mechanism to allow user to create, use, and
  delete loadbalancer in a convinent way.
* introduce some networking resource types(such as bandwidth, iops) so that
  Ingress (and intranet l7 loadbalancing rules which may eventually be added)
  can request "how many networking resource I expected to use"

### Background

#### Current behavior of Ingress
Ingress can be used to expose a service in the kubernetes cluster:

* usually cluster admin deploys one ingress-controller Pod
* user creates Ingress resources
* the ingress-controller Pod will list&watch ***All*** Ingress Resources in the
  cluster
  * on bare-metal, the ingress-controller then call the cloud provider to sync
    the ingress L7 loadbalancing rules
  * on cloud provider case, the ingress-controller then sync the
    nginx(or haproxy, etc) config and reload
* user out of cluster then can access service in the cluster by:
  * on bare-metal, accessing the node's ip on which ingress-controller Pod is
    running, ingress-controller Pod will forward request into cluster based on
    rules defined in Ingress resource
  * on cloud-provider, accessing the ip provided by cloud provider loadbalancer,
    cloud provider will forward request into cluster based on rules defined in
    Ingress Resource

##### Limitations of Ingress
###### On bare-metal only
1. l4 client ip is lost
2. It does not provide High Availability because client needs to know the IP
   addresss of the node where ingress-controller Pod is running. In case of a
   failure the ingress-controller Pod can be moved to a different node

###### Both on bare-metal and on cloud provider
3. How many ingress-controller Pod should run in a cluster? Should all
   ingress-controller Pod list&watch all Ingress Resource with out distinction?
   There is no way to bind or schedule Ingress resource to a ingress-controller
   Pod, which result in:

   * insufficient or excessive use of neworking resource.
   * reload storm when update Ingress resource

2. Ingress resource is actually internet l7 loadbalancing rules, intranet l7
   loadbalancing rules has not been supported yet.


### Goal
##### LoadBalancer
Explicitly expose loadbalancer to users, let Ingress resource(internet
loadbalancing rules) and intranet loadbalancing rules to "use" or "consume"
loadbalancers based on on the networking resources(bandwidth, iops, etc) the
loadbalancing rules request.

*Why "LoadBalancer"*, instead of someting like "IngressService":

Ingress is actually internet l7 loadbalancing rules. Not only Ingress can
"use" or "consume" loadbalancer, intranet l7 loadbalancing rules also can.
Something like "IngressService" is a little confusing.


##### LoadBalancerClaim
* make dynamically provision LoadBalancer possible
* Ingress(and intranet L7 loadbalancing rules) "use" or "consume" a bronze
  loadbalancer by LoadBalancerClaim. A shceduler help to pick a best-matching
  LoadBalancer for LoadBalancerClaim.
* Ingress can "consume" a loadbalancer, as well as loadbalancerclaim.
  Just like the PV/PVC model, Pod can mount PV, as well as PVC.
* for more background see https://github.com/kubernetes/kubernetes/issues/30151

##### Networking resource
Allow user to define how many networking resources(bandwidth, iops, etc) a
Ingress resource request. Then loadbalancer-scheduler can pick a best-matching
one.

##### Loadbalancer controller and Loadbalancer provider
Loadbalancer provider is responssible for:

* provide loadbalancer Create/Delete/Update/Delete interface

Loadbalancer-controller is responsible for:

* list&watch Ingress resources and call Loadbalancer provider to update
  corresponding loadbalancer's loadbalancing rules
* pick a best-matching loadbalancer from existing loadbalancer pool for
  loadbalancerclaim
* call loadbalancer provider to dynamically provision a loadbalancer for
  loadbalancerclaim when it can not find a matching one among existing
  loadbalancer
* recycle or deprovision a loadbalancer when no consumers.


### NoneGoal
* LoadBalancerClass: different type(internet or intranet), different qos level, etc
* LoadBalancer scheduling over-commit

##### networking resource type
```
const (
    ResourceBandWidth ResourceName = "network-bandwidth"
    ResourceIOPS ResourceName = "network-iops"
)
```

We may introduce more networking resource in the future.

##### Loadbalancer API
```
type LoadBalancer struct {
    unversioned.TypeMeta `json:",inline"`
    ObjectMeta           `json:"metadata,omitempty"`

	//Spec defines a loadbalancer owned by the cluster
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
	GCELoadBalancer *GCELoadBalancerSource `json:"gceLoadBalancer,omitempty"`
	AWSLoadBalancer *AWSLoadBalancerSource `json:"awsLoadBalancer,omitempty"`
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
	Type BareMetalLoadBalancerType `json:"type"`
	ServiceName string `json:"serviceName"` //BareMetalLoadBalancer is actually a nginx or haproxy app deploymented in "kube-system" namespace
}

type BareMetalLoadBalancerType string

const (
	NginxLoadBalancer BareMetalLoadBalancerType = "Nginx"
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

##### LoadbalancerClaim API
```
type LoadbalancerClaim struct {
    unversioned.TypeMeta `json:",inline"`
	ObjectMeta           `json:"metadata,omitempty"`

	//Spec defines a loadbalancer owned by the cluster
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

##### change on Ingress API
Add a new knob "LoadBalancer" in IngressSpec, so that Ingress can "use" or
"consume" a LoadBalancer

```
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

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/federation.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
