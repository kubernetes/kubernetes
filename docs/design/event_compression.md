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
[here](http://releases.k8s.io/release-1.0/docs/design/event_compression.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes Event Compression

This document captures the design of event compression.


## Background

Kubernetes components can get into a state where they generate tons of events which are identical except for the timestamp. For example, when pulling a non-existing image, Kubelet will repeatedly generate `image_not_existing` and `container_is_waiting` events until upstream components correct the image. When this happens, the spam from the repeated events makes the entire event mechanism useless. It also appears to cause memory pressure in etcd (see [#3853](https://github.com/GoogleCloudPlatform/kubernetes/issues/3853)).

## Proposal

Each binary that generates events (for example, `kubelet`) should keep track of previously generated events so that it can collapse recurring events into a single event instead of creating a new instance for each new event.

Event compression should be best effort (not guaranteed). Meaning, in the worst case, `n` identical (minus timestamp) events may still result in `n` event entries.

## Design

Instead of a single Timestamp, each event object [contains](http://releases.k8s.io/HEAD/pkg/api/types.go#L1111) the following fields:
 * `FirstTimestamp util.Time`
   * The date/time of the first occurrence of the event.
 * `LastTimestamp util.Time`
   * The date/time of the most recent occurrence of the event.
   * On first occurrence, this is equal to the FirstTimestamp.
 * `Count int`
   * The number of occurrences of this event between FirstTimestamp and LastTimestamp
   * On first occurrence, this is 1.

Each binary that generates events:
 * Maintains a historical record of previously generated events:
   * Implemented with ["Least Recently Used Cache"](https://github.com/golang/groupcache/blob/master/lru/lru.go) in [`pkg/client/record/events_cache.go`](../../pkg/client/record/events_cache.go).
   * The key in the cache is generated from the event object minus timestamps/count/transient fields, specifically the following events fields are used to construct a unique key for an event:
     * `event.Source.Component`
     * `event.Source.Host`
     * `event.InvolvedObject.Kind`
     * `event.InvolvedObject.Namespace`
     * `event.InvolvedObject.Name`
     * `event.InvolvedObject.UID`
     * `event.InvolvedObject.APIVersion`
     * `event.Reason`
     * `event.Message`
   * The LRU cache is capped at 4096 events. That means if a component (e.g. kubelet) runs for a long period of time and generates tons of unique events, the previously generated events cache will not grow unchecked in memory. Instead, after 4096 unique events are generated, the oldest events are evicted from the cache.
 * When an event is generated, the previously generated events cache is checked (see [`pkg/client/record/event.go`](http://releases.k8s.io/HEAD/pkg/client/record/event.go)).
   * If the key for the new event matches the key for a previously generated event (meaning all of the above fields match between the new event and some previously generated event), then the event is considered to be a duplicate and the existing event entry is updated in etcd:
     * The new PUT (update) event API is called to update the existing event entry in etcd with the new last seen timestamp and count.
     * The event is also updated in the previously generated events cache with an incremented count, updated last seen timestamp, name, and new resource version (all required to issue a future event update).
   * If the key for the new event does not match the key for any previously generated event (meaning none of the above fields match between the new event and any previously generated events), then the event is considered to be new/unique and a new event entry is created in etcd:
     * The usual POST/create event API is called to create a new event entry in etcd.
     * An entry for the event is also added to the previously generated events cache.

## Issues/Risks

 * Compression is not guaranteed, because each component keeps track of event history in memory
   * An application restart causes event history to be cleared, meaning event history is not preserved across application restarts and compression will not occur across component restarts.
   * Because an LRU cache is used to keep track of previously generated events, if too many unique events are generated, old events will be evicted from the cache, so events will only be compressed until they age out of the events cache, at which point any new instance of the event will cause a new entry to be created in etcd.

## Example

Sample kubectl output

```console
FIRSTSEEN                         LASTSEEN                          COUNT               NAME                                          KIND                SUBOBJECT                                REASON              SOURCE                                                  MESSAGE
Thu, 12 Feb 2015 01:13:02 +0000   Thu, 12 Feb 2015 01:13:02 +0000   1                   kubernetes-minion-4.c.saad-dev-vms.internal   Minion                                                       starting            {kubelet kubernetes-minion-4.c.saad-dev-vms.internal}   Starting kubelet.
Thu, 12 Feb 2015 01:13:09 +0000   Thu, 12 Feb 2015 01:13:09 +0000   1                   kubernetes-minion-1.c.saad-dev-vms.internal   Minion                                                       starting            {kubelet kubernetes-minion-1.c.saad-dev-vms.internal}   Starting kubelet.
Thu, 12 Feb 2015 01:13:09 +0000   Thu, 12 Feb 2015 01:13:09 +0000   1                   kubernetes-minion-3.c.saad-dev-vms.internal   Minion                                                       starting            {kubelet kubernetes-minion-3.c.saad-dev-vms.internal}   Starting kubelet.
Thu, 12 Feb 2015 01:13:09 +0000   Thu, 12 Feb 2015 01:13:09 +0000   1                   kubernetes-minion-2.c.saad-dev-vms.internal   Minion                                                       starting            {kubelet kubernetes-minion-2.c.saad-dev-vms.internal}   Starting kubelet.
Thu, 12 Feb 2015 01:13:05 +0000   Thu, 12 Feb 2015 01:13:12 +0000   4                   monitoring-influx-grafana-controller-0133o    Pod                                                          failedScheduling    {scheduler }                                            Error scheduling: no minions available to schedule pods
Thu, 12 Feb 2015 01:13:05 +0000   Thu, 12 Feb 2015 01:13:12 +0000   4                   elasticsearch-logging-controller-fplln        Pod                                                          failedScheduling    {scheduler }                                            Error scheduling: no minions available to schedule pods
Thu, 12 Feb 2015 01:13:05 +0000   Thu, 12 Feb 2015 01:13:12 +0000   4                   kibana-logging-controller-gziey               Pod                                                          failedScheduling    {scheduler }                                            Error scheduling: no minions available to schedule pods
Thu, 12 Feb 2015 01:13:05 +0000   Thu, 12 Feb 2015 01:13:12 +0000   4                   skydns-ls6k1                                  Pod                                                          failedScheduling    {scheduler }                                            Error scheduling: no minions available to schedule pods
Thu, 12 Feb 2015 01:13:05 +0000   Thu, 12 Feb 2015 01:13:12 +0000   4                   monitoring-heapster-controller-oh43e          Pod                                                          failedScheduling    {scheduler }                                            Error scheduling: no minions available to schedule pods
Thu, 12 Feb 2015 01:13:20 +0000   Thu, 12 Feb 2015 01:13:20 +0000   1                   kibana-logging-controller-gziey               BoundPod            implicitly required container POD        pulled              {kubelet kubernetes-minion-4.c.saad-dev-vms.internal}   Successfully pulled image "kubernetes/pause:latest"
Thu, 12 Feb 2015 01:13:20 +0000   Thu, 12 Feb 2015 01:13:20 +0000   1                   kibana-logging-controller-gziey               Pod                                                          scheduled           {scheduler }                                            Successfully assigned kibana-logging-controller-gziey to kubernetes-minion-4.c.saad-dev-vms.internal
```

This demonstrates what would have been 20 separate entries (indicating scheduling failure) collapsed/compressed down to 5 entries.

## Related Pull Requests/Issues

 * Issue [#4073](https://github.com/GoogleCloudPlatform/kubernetes/issues/4073): Compress duplicate events
 * PR [#4157](https://github.com/GoogleCloudPlatform/kubernetes/issues/4157): Add "Update Event" to Kubernetes API
 * PR [#4206](https://github.com/GoogleCloudPlatform/kubernetes/issues/4206): Modify Event struct to allow compressing multiple recurring events in to a single event
 * PR [#4306](https://github.com/GoogleCloudPlatform/kubernetes/issues/4306): Compress recurring events in to a single event to optimize etcd storage
 * PR [#4444](https://github.com/GoogleCloudPlatform/kubernetes/pull/4444): Switch events history to use LRU cache instead of map


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/event_compression.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
