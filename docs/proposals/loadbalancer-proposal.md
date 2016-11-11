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
Thie proposal aims to:

* introduce a first class API "LoadBalancer" so that user can
  create/update/delete a loadbalancer.
* introduce a first class API "LoadBalancerClaim" so that loadbalancer can be
  dynamically provisioned and "used" based on networking resource request.
* introduce some networking resource such as bandwidth and iops, so that Ingress
  can declare how many networking resource it request, and let k8s to schedule a
  loadbalancer for it.


### Background

#### Current behavior of Ingress
Ingress can be used to expose a service in the kubernetes cluster:

* usually cluster admin deploys one ingress-controller Pod
* user creates Ingress resources
* the ingress-controller Pod will list&watch ***All*** Ingress Resources in the
  cluster
  * on cloud provider, the ingress-controller then call the cloud provider to
    sync the ingress L7 loadbalancing rules
  * on bare-metal  case, the ingress-controller then sync the
    nginx(or haproxy, etc) config and reload
* user out of cluster then can access service in the cluster by:
  * on bare-metal, accessing the node's ip on which ingress-controller Pod is
    running, ingress-controller Pod will forward request into cluster based on
    rules defined in Ingress resource
  * on cloud-provider, accessing the ip provided by cloud provider loadbalancer,
    cloud provider will forward request into cluster based on rules defined in
    Ingress Resource

##### Limitations of current Ingress implementation
* On bare-metal, it does not provide High Availability because client needs
  to know the IP addresss of the node where ingress-controller Pod is running.
  In case of a failure the ingress-controller Pod can be moved to a different
  node.
* How many ingress-controller Pod should run in a cluster? Should all
  ingress-controller Pod list&watch all Ingress resource with out distinction?
  There is no way to bind or schedule Ingress resource to a ingress-controller
  Pod, which result in:

   * insufficient or excessive use of neworking resource.
   * reload storm when update Ingress resource

* Ingress resource is actually internet l7 loadbalancing rules, intranet l7
  loadbalancing rules has not been supported yet. Eventually, We need a general
  mechanism for both the Ingress and intranet L7 lb rules "consume" a
  loadbalancer


### Goal and NoneGoal
##### Goal
* LoadBalancer API
* LoadBalancerClaim API
* networking resource
* loadbalancer provider
* loadbalancer scheduling

##### NoGoal
* LoadBalancer HA on bare-metal
* LoadBalancerClass: different type(internet or intranet), different qos level, etc
* LoadBalancer scheduling over-commitment

### Design
##### LoadBalancer
Introduce "LoadBalancer" as first-class API, let Ingress resource (internet
loadbalancing rules) and intranet loadbalancing rules to "use" or "consume"
loadbalancers based on on the networking resources(bandwidth, iops, etc) they
request. Just like how Pod "consume" Node.



##### LoadBalancerClaim

Introduce "LoadBalancerClaim" as first-class API so that:

* Ingress(and intranet L7 loadbalancing rules) can "use" or "consume" a bronze
loadbalancer by "using" or "consuming" a LoadBalancerClaim, just like how
Pod mount a PV by mounting PVC.
* make dynamically loadbalancer provisioning possible, just like how PV is
  dynamically provisioned.

for more background see https://github.com/kubernetes/kubernetes/issues/30151

##### LoadBalancer provider interface

* provide loadbalancer Create/Update/Delete interface
* k8s can have multiple LoadBalancer provider implementation, such as:
  * AWS loadbalancer provider
  * GCE loadbalancer provider
  * bare-metal nginx loadbalancer provider
  * bare-metal haproxy loadbalancer provider

##### Loadbalancer-controller

loadbalancer-controller is responsible for:

* list&watch Ingress resources and call Loadbalancer provider to update
  corresponding loadbalancer's loadbalancing rules
* pick a best-matching loadbalancer from existing loadbalancer pool for
  loadbalancerclaim based on the networking resource the loadbalancerclaim
  request
* call loadbalancer provider to dynamically provision a loadbalancer for
  loadbalancerclaim when it can not find a matching one among existing
  loadbalancer
* recycle or deprovision a loadbalancer when no consumers.

##### Networking resource
Allow user to define how many networking resources(bandwidth, iops, etc) an
Ingress request. Then loadbalancer-controller can pick a best-matching one.

##### Loadbalancer scheduling
Unlike PV/PVC, LB and LBC binding relationship is not exclusive, which means
multiple LBC can bind to one LB. For example, if we have a loadbalancer with
3G bandwidth, we can bind 6 LBC each request 500m bandwidth on it. Some
scheduling mechanism is needed in such a case.

If we go further, we may eventually introduce the request/limit model for
networking resource so that we can over commit.


### API

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

### Implementation

#### loadbalancer

The current loadbalancer(ingress-controller) implementation has some
limitations on cloud provider when we have multiple loadbalancers:

* we need deploy multiple ingress-controller pod even on cloud provider, it
  result in some excessive use of resource. Actually it fine that a cluster
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

### Implementation plan

##### First step: make it workable
* implement the scheduling, provisioning and recycling logic first.
* ingress-controller just works as currently, but it just list&watch ingresses
  assigned to it, insteal all ingresses

##### Second step: loadbalancer provider
* Add loadbalancer provider in loadbalancer-controller
* refactor the current ingress-controller implementation as I descrive in the
  "Implementation" section and rename "ingress-controller" as
  "nginx-loadbalancer"

##### Long term
* loadbalancer scheduling over-commitment


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/federation.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
