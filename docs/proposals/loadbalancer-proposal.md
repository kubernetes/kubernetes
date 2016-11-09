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

* explicitly expose cloud loadbalancer to users
* introduce `LoadbalancerClaim` mechanism to allow user to create, use, and delete
loadbalancer in a convienet way.

####  Goal
* LoadBalancer and LoadBalancerClaim API
* LoadBalancer provisioning
* LoadBalancer scheduling based on bandwidth and iops request

#### NoneGoal
* LoadBalancerClass: different type(internet or intranet), different qos level etc
* LoadBalancer scheduling over-commit


### Design
* Add a `LoadBalancer` API to expose LoadBalancer explicitly to user
* Add bandwidth and iops as new resource type, to describe the resource capacity
  of a LoadBalancer and describe the resource a LoadBalancerClaim request
* When user create a Ingress resource, we need a LoadBalancer to satisfy the
  request, but which LoadBalancer to choose? we propose that Ingress "use" a
  loadbalancer indirectly through LoadBalancerClaim
* If use know exactly which LoadBalancer to use, he can create a LoadBalancerClaim
  and bind it with the LoadBalancer manully. Otherwise, a loadbalancer-scheduler
  will help user to make the decision based on the resource a LoadBalancerClaim
  request and LoadBalancer's remaining capacity
* LoadBalancerClaim->LoadBalancer binding is non-exclusive

### Implement
* Add a loadbalancer-scheduler to schedule loadbalancerclaim to loadbalancer
* Add a loadbalancer-cliam controller to:
    * dynamically provison lb when scheduler can not find a suitable loadbalancer
      for a loadbalancerclaim
    * recycle lb when ALL lbc bound to it has been deleted

* Redesign the current ingress-controller, as follows:
    * addd some loadbalancer related function in cloud provider interface
    * add a loadbalancer controller to list&watch all Ingress resource and
      call cloud provider to sync ingress rules, instead of ingress-controller
      pod list&watch Ingress resource itself
    * for bare-metal cluster, write a fake cloud provider, loadbalancer-controller
      call cloud provider, then cloud provider call the nginx/haproxy pod to
      update and reload config.

### API

##### networking resource type
``` go
const (
    ResourceBandWidth ResourceName = "network-bandwidth"
    ResourceIOPS ResourceName = "network-iops"
)
```

##### Loadbalancer API
``` go
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
	// used for LoadBalancers that failed to be correctly deprovisioned after
	// all LoadBalancerClaim boun with it has been deleted
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
``` go
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

##### Ingress
* Add a new field "ClaimName" in IngressSpec, so that Ingress Resource can "use"
  a LoadBalancer


``` go
// IngressSpec describes the Ingress the user wishes to exist.
type IngressSpec struct {
    /*
    ...
    */
    // ClaimName is the name of a LoadBalancerClaim in the same namespace as
    // the Ingress using this cliam
    ClaimName string `json:"claimName"`
}

```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/federation.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
