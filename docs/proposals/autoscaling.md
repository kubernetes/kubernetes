<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

---

# WARNING:

## This document is outdated. It is superseded by [the horizontal pod autoscaler design doc](../design/horizontal-pod-autoscaler.md).

---

## Abstract

Auto-scaling is a data-driven feature that allows users to increase or decrease capacity as needed by controlling the
number of pods deployed within the system automatically.

## Motivation

Applications experience peaks and valleys in usage.  In order to respond to increases and decreases in load, administrators
scale their applications by adding computing resources.  In the cloud computing environment this can be
done automatically based on statistical analysis and thresholds.

### Goals

* Provide a concrete proposal for implementing auto-scaling pods within Kubernetes
* Implementation proposal should be in line with current discussions in existing issues:
    * Scale verb - [1629](http://issue.k8s.io/1629)
    * Config conflicts - [Config](https://github.com/kubernetes/kubernetes/blob/c7cb991987193d4ca33544137a5cb7d0292cf7df/docs/config.md#automated-re-configuration-processes)
    * Rolling updates - [1353](http://issue.k8s.io/1353)
    * Multiple scalable types - [1624](http://issue.k8s.io/1624)

## Constraints and Assumptions

* This proposal is for horizontal scaling only.  Vertical scaling will be handled in [issue 2072](http://issue.k8s.io/2072)
* `ReplicationControllers` will not know about the auto-scaler, they are the target of the auto-scaler.  The `ReplicationController` responsibilities are
constrained to only ensuring that the desired number of pods are operational per the [Replication Controller Design](../user-guide/replication-controller.md#responsibilities-of-the-replication-controller)
* Auto-scalers will be loosely coupled with data gathering components in order to allow a wide variety of input sources
* Auto-scalable resources will support a scale verb ([1629](http://issue.k8s.io/1629))
such that the auto-scaler does not directly manipulate the underlying resource.
* Initially, most thresholds will be set by application administrators. It should be possible for an autoscaler to be
written later that sets thresholds automatically based on past behavior (CPU used vs incoming requests).
* The auto-scaler must be aware of user defined actions so it does not override them unintentionally (for instance someone
explicitly setting the replica count to 0 should mean that the auto-scaler does not try to scale the application up)
* It should be possible to write and deploy a custom auto-scaler without modifying existing auto-scalers
* Auto-scalers must be able to monitor multiple replication controllers while only targeting a single scalable
object (for now a ReplicationController, but in the future it could be a job or any resource that implements scale)

## Use Cases

### Scaling based on traffic

The current, most obvious, use case is scaling an application based on network traffic like requests per second.  Most
applications will expose one or more network endpoints for clients to connect to. Many of those endpoints will be load
balanced or situated behind a proxy - the data from those proxies and load balancers can be used to estimate client to
server traffic for applications. This is the primary, but not sole, source of data for making decisions.

Within Kubernetes a [kube proxy](../user-guide/services.md#ips-and-vips)
running on each node directs service requests to the underlying implementation.

While the proxy provides internal inter-pod connections, there will be L3 and L7 proxies and load balancers that manage
traffic to backends. OpenShift, for instance, adds a "route" resource for defining external to internal traffic flow.
The "routers" are HAProxy or Apache load balancers that aggregate many different services and pods and can serve as a
data source for the number of backends.

### Scaling based on predictive analysis

Scaling may also occur based on predictions of system state like anticipated load, historical data, etc.  Hand in hand
with scaling based on traffic, predictive analysis may be used to determine anticipated system load and scale the application automatically.

### Scaling based on arbitrary data

Administrators may wish to scale the application based on any number of arbitrary data points such as job execution time or
duration of active sessions.  There are any number of reasons an administrator may wish to increase or decrease capacity which
means the auto-scaler must be a configurable, extensible component.

## Specification

In order to facilitate talking about auto-scaling the following definitions are used:

* `ReplicationController` - the first building block of auto scaling.  Pods are deployed and scaled by a `ReplicationController`.
* kube proxy - The proxy handles internal inter-pod traffic, an example of a data source to drive an auto-scaler
* L3/L7 proxies - A routing layer handling outside to inside traffic requests, an example of a data source to drive an auto-scaler
* auto-scaler - scales replicas up and down by using the `scale` endpoint provided by scalable resources (`ReplicationController`)


### Auto-Scaler

The Auto-Scaler is a state reconciler responsible for checking data against configured scaling thresholds
and calling the `scale` endpoint to change the number of replicas.  The scaler will
use a client/cache implementation to receive watch data from the data aggregators and respond to them by
scaling the application.  Auto-scalers are created and defined like other resources via REST endpoints and belong to the
namespace just as a `ReplicationController` or `Service`.

Since an auto-scaler is a durable object it is best represented as a resource.

```go
    //The auto scaler interface
    type AutoScalerInterface interface {
        //ScaleApplication adjusts a resource's replica count.  Calls scale endpoint.  
        //Args to this are based on what the endpoint
        //can support.  See http://issue.k8s.io/1629
        ScaleApplication(num int) error
    }

    type AutoScaler struct {
        //common construct
        TypeMeta
        //common construct
        ObjectMeta

        //Spec defines the configuration options that drive the behavior for this auto-scaler
        Spec    AutoScalerSpec  

        //Status defines the current status of this auto-scaler.
        Status  AutoScalerStatus
     }

    type AutoScalerSpec struct {
        //AutoScaleThresholds holds a collection of AutoScaleThresholds that drive the auto scaler
        AutoScaleThresholds []AutoScaleThreshold

        //Enabled turns auto scaling on or off
        Enabled boolean

        //MaxAutoScaleCount defines the max replicas that the auto scaler can use.  
        //This value must be greater than 0 and >= MinAutoScaleCount
        MaxAutoScaleCount int

        //MinAutoScaleCount defines the minimum number replicas that the auto scaler can reduce to,
        //0 means that the application is allowed to idle
        MinAutoScaleCount int

        //TargetSelector provides the scalable target(s).  Right now this is a ReplicationController
        //in the future it could be a job or any resource that implements scale.  
        TargetSelector map[string]string

        //MonitorSelector defines a set of capacity that the auto-scaler is monitoring
        //(replication controllers).  Monitored objects are used by thresholds to examine
        //statistics.  Example: get statistic X for object Y to see if threshold is passed
        MonitorSelector map[string]string
    }

    type AutoScalerStatus struct {
        // TODO: open for discussion on what meaningful information can be reported in the status
        // The status may return the replica count here but we may want more information
        // such as if the count reflects a threshold being passed
    }


     //AutoScaleThresholdInterface abstracts the data analysis from the auto-scaler
     //example: scale by 1 (Increment) when RequestsPerSecond (Type) pass
     //comparison (Comparison) of 50 (Value) for 30 seconds (Duration)
     type AutoScaleThresholdInterface interface {
        //called by the auto-scaler to determine if this threshold is met or not
        ShouldScale() boolean
     }


     //AutoScaleThreshold is a single statistic used to drive the auto-scaler in scaling decisions
     type AutoScaleThreshold struct {
        // Type is the type of threshold being used, intention or value
        Type AutoScaleThresholdType

        // ValueConfig holds the config for value based thresholds
        ValueConfig AutoScaleValueThresholdConfig

        // IntentionConfig holds the config for intention based thresholds
        IntentionConfig AutoScaleIntentionThresholdConfig
     }

     // AutoScaleIntentionThresholdConfig holds configuration for intention based thresholds
     // a intention based threshold defines no increment, the scaler will adjust by 1 accordingly
     // and maintain once the intention is reached.  Also, no selector is defined, the intention
     // should dictate the selector used for statistics.  Same for duration although we
     // may want a configurable duration later so intentions are more customizable.
     type AutoScaleIntentionThresholdConfig struct {
        // Intent is the lexicon of what intention is requested
        Intent  AutoScaleIntentionType

        // Value is intention dependent in terms of above, below, equal and represents
        // the value to check against
        Value   float
     }

     // AutoScaleValueThresholdConfig holds configuration for value based thresholds
     type AutoScaleValueThresholdConfig struct {
        //Increment determines how the auot-scaler should scale up or down (positive number to
        //scale up based on this threshold negative number to scale down by this threshold)
        Increment int
        //Selector represents the retrieval mechanism for a statistic value from statistics
        //storage.  Once statistics are better defined the retrieval mechanism may change.
        //Ultimately, the selector returns a representation of a statistic that can be
        //compared against the threshold value.  
        Selector map[string]string
        //Duration is the time lapse after which this threshold is considered passed
        Duration time.Duration
        //Value is the number at which, after the duration is passed, this threshold is considered
        //to be triggered
        Value float
        //Comparison component to be applied to the value.
        Comparison string
     }

     // AutoScaleThresholdType is either intention based or value based
     type AutoScaleThresholdType string

     // AutoScaleIntentionType is a lexicon for intentions such as "cpu-utilization",
     // "max-rps-per-endpoint"
     type AutoScaleIntentionType string
```

#### Boundary Definitions

The `AutoScaleThreshold` definitions provide the boundaries for the auto-scaler.  By defining comparisons that form a range
along with positive and negative increments you may define bi-directional scaling.  For example the upper bound may be
specified as "when requests per second rise above 50 for 30 seconds scale the application up by 1" and a lower bound may
be specified as "when requests per second fall below 25 for 30 seconds scale the application down by 1 (implemented by using -1)".

### Data Aggregator

This section has intentionally been left empty.  I will defer to folks who have more experience gathering and analyzing
time series statistics.

Data aggregation is opaque to the auto-scaler resource.  The auto-scaler is configured to use `AutoScaleThresholds`
that know how to work with the underlying data in order to know if an application must be scaled up or down.   Data aggregation
must feed a common data structure to ease the development of `AutoScaleThreshold`s but it does not matter to the
auto-scaler whether this occurs in a push or pull implementation, whether or not the data is stored at a granular level,
or what algorithm is used to determine the final statistics value.  Ultimately, the auto-scaler only requires that a statistic
resolves to a value that can be checked against a configured threshold.

Of note: If the statistics gathering mechanisms can be initialized with a registry other components storing statistics can
potentially piggyback on this registry.

### Multi-target Scaling Policy

If multiple scalable targets satisfy the `TargetSelector` criteria the auto-scaler should be configurable as to which
target(s) are scaled.  To begin with, if multiple targets are found the auto-scaler will scale the largest target up
or down as appropriate.  In the future this may be more configurable.

### Interactions with a deployment

In a deployment it is likely that multiple replication controllers must be monitored.  For instance, in a [rolling deployment](../user-guide/replication-controller.md#rolling-updates)
there will be multiple replication controllers, with one scaling up and another scaling down.  This means that an
auto-scaler must be aware of the entire set of capacity that backs a service so it does not fight with the deployer.  `AutoScalerSpec.MonitorSelector`
is what provides this ability.  By using a selector that spans the entire service the auto-scaler can monitor capacity
of multiple replication controllers and check that capacity against the `AutoScalerSpec.MaxAutoScaleCount` and
`AutoScalerSpec.MinAutoScaleCount` while still only targeting a specific set of `ReplicationController`s with `TargetSelector`.

In the course of a deployment it is up to the deployment orchestration to decide how to manage the labels
on the replication controllers if it needs to ensure that only specific replication controllers are targeted by
the auto-scaler.  By default, the auto-scaler will scale the largest replication controller that meets the target label
selector criteria.

During deployment orchestration the auto-scaler may be making decisions to scale its target up or down.  In order to prevent
the scaler from fighting with a deployment process that is scaling one replication controller up and scaling another one
down the deployment process must assume that the current replica count may be changed by objects other than itself and
account for this in the scale up or down process.   Therefore, the deployment process may no longer target an exact number
of instances to be deployed.  It must be satisfied that the replica count for the deployment meets or exceeds the number
of requested instances.

Auto-scaling down in a deployment scenario is a special case.  In order for the deployment to complete successfully the
deployment orchestration must ensure that the desired number of instances that are supposed to be deployed has been met.
If the auto-scaler is trying to scale the application down (due to no traffic, or other statistics) then the deployment
process and auto-scaler are fighting to increase and decrease the count of the targeted replication controller.  In order
to prevent this, deployment orchestration should notify the auto-scaler that a deployment is occurring.  This will
temporarily disable negative decrement thresholds until the deployment process is completed.  It is more important for
an auto-scaler to be able to grow capacity during a deployment than to shrink the number of instances precisely.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/autoscaling.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
