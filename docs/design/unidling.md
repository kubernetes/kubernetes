# Unidling Pods
## Abstract

Unidling is an event-driven feature that allows users to automatically create destination pods for
an idled service when a connection is made to it.

## Motivation

Not all pods in a Kubernetes cluster will be active at all times.  When all of the pods
resolving to a service are inactive, new ones must be created when a new request for that service
is received.  

An idled service is a service with zero endpoints.  The process of creating new pods for an idled
service in response to a new request to that service is called "unidling".  This proposal covers
how unidling should be performed in Kubernetes and how external proxies and routing components
can trigger unidling.

Idling is the process of determining which pods to stop and stopping them. Idling will be dealt
with in a separate proposal and is out of scope here.  

## Relation to Autoscaling

Unidling is closely related to the problem of autoscaling
(see [#2863](https://github.com/GoogleCloudPlatform/kubernetes/pull/2863)). This proposal addresses
unidling in an isolated way to keep the problem scope manageable.  The hypothetical unidler daemon
in this proposal could easily be the same as the auto-scaler controller proposed in 
[#2863](https://github.com/GoogleCloudPlatform/kubernetes/pull/2863).  This proposal is meant to
call out problems specific to unidling rather than to be a prescriptive description of a single
proposed implementation.

## Constraints and Assumptions

- No specific idling mechanism or autoscaler is assumed
- The case where a service is backed by pods which are not managed by a replication controller: 
  - Not covered by this proposal at all
  - Requests to unidle these service should be gracefully ignored by components scoped to this
    proposal
  - A separate unidler may be written for these

## Use Cases

The following use cases are explored by this proposal:

1.  As a cluster operator, I want new request to an idled service to trigger the creation of new
    destination pods for that service
2.  As a cluster operator, I want requests to an idled service to be buffered while the service is
    being unidled
3.  As a cluster operator, I want the system to be able to tolerate bursts of requests to unidle a
    service without thrashing

## Design

The requirements for unidling an idle service S with replication controller R are as follows:

1.  The first request to S after it has become idle must:
    1.  trigger unidling of S
    2.  be buffered (timeout-bounded) until a destination pod is available
2.  R must be resized to *n* >= 1
3.  Subsequent requests to S while S is being unidled must:
    1.  Not cause thrashing
    2.  Be buffered (timeout-bounded) until a destination pod is available

### Where does unidle behavior live?

A loose coupling between the triggers of unidling and the unidling behavior itself is desirable.
The best way to acheive this loose coupling is to have any system component that wants to trigger
an unidle to delegate performing it to another component.

Receiving a request for an idled service can be modeled as an api Event for that service.  The
fields of the Event would be set as follows:

- `Condition`: should always be `WantPods`; unidler reconcilers will filter their watches on this
  field value
- `InvolvedObject`: an `ObjectReference` to the service to be unidled

This is the minimum amount of information required; as unidler implementations become more 
sophisticated there may be additional information carried by the event such as the time the event
creator will wait for pods to be created before returning a 503 to the requester.

### Triggering unidle

A system component that wishes to trigger unidle of a pod, such as the kube-proxy, signals that a
pod should be unidled by creating `WantPods` event to signal to the unidler that a pod should be
unidled.  It is the responsibility of the signaling component to handle the user request while the
pod is unidled, implement timeouts, etc.  

For example, the kube-proxy respond to a request to an idled pod as follows:

1.  Create a `WantPods` event for S
2.  Block until the kube-proxy receives endpoint information for S or a timeout is reached
3.  If endpoints are created before the timeout, dispatch requests to S
4.  If the timeout elapses before endpoints are created, return a 503 to the requester

### 

### The unidler and its algorithm

A new default controller will be introduced to watch for `WantPods` events.  This controller will
implement the behavior for unidling and be responsible for preventing thrashing when multiple
`WantPods` events are received.

The unidler controller is responsible for determining how to unidle a service.  Initially we will
implement a simple annotation-based policy to resolve services to replication controllers for
unidling that doesn't require introduction of a new resource to configure the unidler.  In the
future, more sophisticated unidlers may be configured using a new API resource.

In the first iteration the algorithm to unidle a service S will be:

1.  Attempt to resolve S to a replication controller R that is annotated as unidling S
2.  For each R found, ensure that the replica count is zero
	1.  If there is an R with *n* >= 1, ignore the request
3.  Resize the first R to *n* >= 1

If no R is found no action is taken for the event to allow another controller to handle it.

### Example: single kube-proxy receives single request for idled service

This simplest possible unidle case is where an idled service S is unidled by a single request on a
single node N.  

1.  The kube-proxy on node N receives a request for S
2.  The kube-proxy socket handler for S creates a `WantPods` event and waits X seconds for new
    endpoints for S to be created
3.  The unidler receives a watch event for the `WantPods` event
4.  The unidler resolves the S to some replication controller R
5.  The unidler resizes R to *n* >= 1
6.  The replication controller manager schedules a new pod P for R
7.  The scheduler determines placement and creates a `Binding` for P
8.  The kubelet on the appropriate node processes the `Binding` and runs P
9.  The endpoints controller determines that S has a new endpoint and updates S's endpoints
10. The kube-proxy receives a watch event with the new endpoint information and updates its routing
    table
11. The kube-proxy services the request to the new endpoint

### Thrash prevention

The unidler must be able to correctly handle bursts of `WantPods` events for the same service.
This implies that the unidler has an internal representation of which services have recently been
unidled and that the unidle process itself has checks to ensure that only the minimum number of
pods to unidle a service are created.  Concerns about scaling upon unidle should be handled
exclusively by the autoscaler.

### Example: multiple kube-proxies receive requests for idled service

Thrash prevention becomes relevant when multiple kube-proxies receive requests for an idled service
at approximately the same time.

1.  The kube-proxy on node N1...Nx receive requests for S
2.  The kube-proxy socket handlers create `WantPods` events and wait X seconds for new
    endpoints for S to be created
3.  The unidler receives the first watch event for the `WantPods` event
4.  The unidler resolves the S to some replication controller R
5.  The unidler resizes R to *n* >= 1
6.  The unidler receives subsequent `WantPods` events from other nodes and ignores them since
    there is already a replication controller that resolves to S with *n* >= 1
7.  The replication controller manager schedules a new pod P for R
8.  The scheduler determines placement and creates a `Binding` for P
9.  The kubelet on the appropriate node processes the `Binding` and runs P
10. The endpoints controller determines that S has a new endpoint and updates S's endpoints
11. The kube-proxies receive watch events with the new endpoint information and updates their routing
    tables
12. The kube-proxies service the requests to the new endpoint

## Proposed Design

### Modifications to kube-proxy

The kube-proxy must be modified create events when idled services are requested and make an attempt
to wait until endpoints are available.  Currently, the proxy delegates across a `LoadBalancer`
interface to get the endpoint to proxy a connection to.  It makes sense to detect idled services
inside this interface boundary.

Currently, there is only one implementation of the `LoadBalancer` interface but we do not want to
tie unidle functionality to that implementation.  There should be a new internal component of the
proxier that can be used by any `LoadBalancer` implementation that will handle creating a
`WantPods` event and handle waiting until a timeout for destination pods to become available.  The
fields of the `WantPods` event will be set as described above.

### The `UnidlerController`

The `UnidlerController` will be added to `pkg/unidle`.  It will use the following algorithm to
resolve a service S to replication controllers for unidling:

1.  If there are any replication controllers in S's namespace with the annotation
    `unidle.service=S`, use the first in the list
2.  Otherwise, if there is a replication controller in S's namespace with a `Name` that matches S's
    `Name`, use it

The `UnidlerController` will handle bursts of `WantPods` events by maintaining an internal state
which is essentially a ttl-bounded set. The controller's event handler will implement the following
logic for each `WantPods` event for some service X:

1.  If the key for S is in the cache, drop the event
2.  If the key for S is not in the cache
    1.  Insert the key for S into the cache
    2.  Resolve S to a replication controller R
        1.  If some R is found, resize it to *n* >= 1

Note that the above approach accomodates bursts for services which are backed by pods not managed
by a replication controller.  They are gracefully ignored so that some other controller may handle
them.
