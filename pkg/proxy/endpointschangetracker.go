/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package proxy

import (
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy/metrics"
)

// EndpointsChangeTracker carries state about uncommitted changes to an arbitrary number of
// Endpoints, keyed by their namespace and name.
type EndpointsChangeTracker struct {
	// lock protects lastChangeTriggerTimes
	lock sync.Mutex

	// processEndpointsMapChange is invoked by the apply function on every change.
	// This function should not modify the EndpointsMaps, but just use the changes for
	// any Proxier-specific cleanup.
	processEndpointsMapChange processEndpointsMapChangeFunc

	// addressType is the type of EndpointSlice this proxy tracks
	addressType discovery.AddressType

	// endpointSliceCache holds a simplified version of endpoint slices.
	endpointSliceCache *EndpointSliceCache

	// lastChangeTriggerTimes maps from the Service's NamespacedName to the times of
	// the triggers that caused its EndpointSlice objects to change. Used to calculate
	// the network-programming-latency metric.
	lastChangeTriggerTimes map[types.NamespacedName][]time.Time
	// trackerStartTime is the time when the EndpointsChangeTracker was created, so
	// we can avoid generating network-programming-latency metrics for changes that
	// occurred before that.
	trackerStartTime time.Time
}

type makeEndpointFunc func(info *BaseEndpointInfo, svcPortName *ServicePortName) Endpoint
type processEndpointsMapChangeFunc func(oldEndpointsMap, newEndpointsMap EndpointsMap)

// NewEndpointsChangeTracker initializes an EndpointsChangeTracker
func NewEndpointsChangeTracker(ipFamily v1.IPFamily, hostname string, makeEndpointInfo makeEndpointFunc, processEndpointsMapChange processEndpointsMapChangeFunc) *EndpointsChangeTracker {
	addressType := discovery.AddressTypeIPv4
	if ipFamily == v1.IPv6Protocol {
		addressType = discovery.AddressTypeIPv6
	}

	return &EndpointsChangeTracker{
		addressType:               addressType,
		lastChangeTriggerTimes:    make(map[types.NamespacedName][]time.Time),
		trackerStartTime:          time.Now(),
		processEndpointsMapChange: processEndpointsMapChange,
		endpointSliceCache:        NewEndpointSliceCache(hostname, makeEndpointInfo),
	}
}

// EndpointSliceUpdate updates the EndpointsChangeTracker by adding/updating or removing
// endpointSlice (depending on removeSlice). It returns true if this update contained a
// change that needs to be synced; note that this is different from the return value of
// ServiceChangeTracker.Update().
func (ect *EndpointsChangeTracker) EndpointSliceUpdate(endpointSlice *discovery.EndpointSlice, removeSlice bool) bool {
	if endpointSlice.AddressType != ect.addressType {
		klog.V(4).InfoS("Ignoring unsupported EndpointSlice", "endpointSlice", klog.KObj(endpointSlice), "type", endpointSlice.AddressType, "expected", ect.addressType)
		return false
	}

	namespacedName, _, err := endpointSliceCacheKeys(endpointSlice)
	if err != nil {
		klog.InfoS("Error getting endpoint slice cache keys", "err", err)
		return false
	}

	metrics.EndpointChangesTotal.Inc()

	ect.lock.Lock()
	defer ect.lock.Unlock()

	changeNeeded := ect.endpointSliceCache.updatePending(endpointSlice, removeSlice)

	if changeNeeded {
		metrics.EndpointChangesPending.Inc()
		// In case of Endpoints deletion, the LastChangeTriggerTime annotation is
		// by-definition coming from the time of last update, which is not what
		// we want to measure. So we simply ignore it in this cases.
		// TODO(wojtek-t, robscott): Address the problem for EndpointSlice deletion
		// when other EndpointSlice for that service still exist.
		if removeSlice {
			delete(ect.lastChangeTriggerTimes, namespacedName)
		} else if t := getLastChangeTriggerTime(endpointSlice.Annotations); !t.IsZero() && t.After(ect.trackerStartTime) {
			ect.lastChangeTriggerTimes[namespacedName] =
				append(ect.lastChangeTriggerTimes[namespacedName], t)
		}
	}

	return changeNeeded
}

// checkoutChanges returns a map of pending endpointsChanges and marks them as
// applied.
func (ect *EndpointsChangeTracker) checkoutChanges() map[types.NamespacedName]*endpointsChange {
	metrics.EndpointChangesPending.Set(0)

	return ect.endpointSliceCache.checkoutChanges()
}

// checkoutTriggerTimes applies the locally cached trigger times to a map of
// trigger times that have been passed in and empties the local cache.
func (ect *EndpointsChangeTracker) checkoutTriggerTimes(lastChangeTriggerTimes *map[types.NamespacedName][]time.Time) {
	ect.lock.Lock()
	defer ect.lock.Unlock()

	for k, v := range ect.lastChangeTriggerTimes {
		prev, ok := (*lastChangeTriggerTimes)[k]
		if !ok {
			(*lastChangeTriggerTimes)[k] = v
		} else {
			(*lastChangeTriggerTimes)[k] = append(prev, v...)
		}
	}
	ect.lastChangeTriggerTimes = make(map[types.NamespacedName][]time.Time)
}

// getLastChangeTriggerTime returns the time.Time value of the
// EndpointsLastChangeTriggerTime annotation stored in the given endpoints
// object or the "zero" time if the annotation wasn't set or was set
// incorrectly.
func getLastChangeTriggerTime(annotations map[string]string) time.Time {
	// TODO(#81360): ignore case when Endpoint is deleted.
	if _, ok := annotations[v1.EndpointsLastChangeTriggerTime]; !ok {
		// It's possible that the Endpoints object won't have the
		// EndpointsLastChangeTriggerTime annotation set. In that case return
		// the 'zero value', which is ignored in the upstream code.
		return time.Time{}
	}
	val, err := time.Parse(time.RFC3339Nano, annotations[v1.EndpointsLastChangeTriggerTime])
	if err != nil {
		klog.ErrorS(err, "Error while parsing EndpointsLastChangeTriggerTimeAnnotation",
			"value", annotations[v1.EndpointsLastChangeTriggerTime])
		// In case of error val = time.Zero, which is ignored in the upstream code.
	}
	return val
}

// endpointsChange contains all changes to endpoints that happened since proxy
// rules were synced.  For a single object, changes are accumulated, i.e.
// previous is state from before applying the changes, current is state after
// applying the changes.
type endpointsChange struct {
	previous EndpointsMap
	current  EndpointsMap
}

// UpdateEndpointsMapResult is the updated results after applying endpoints changes.
type UpdateEndpointsMapResult struct {
	// UpdatedServices lists the names of all services with added/updated/deleted
	// endpoints since the last Update.
	UpdatedServices sets.Set[types.NamespacedName]
	// ConntrackCleanupRequired will be true if any UDP ServicePort changed endpoints, false otherwise.
	// It's used to minimise conntrack cleanup calls.
	ConntrackCleanupRequired bool
	// List of the trigger times for all endpoints objects that changed. It's used to export the
	// network programming latency.
	// NOTE(oxddr): this can be simplified to []time.Time if memory consumption becomes an issue.
	LastChangeTriggerTimes map[types.NamespacedName][]time.Time
}

// EndpointsMap maps a service name to a list of all its Endpoints.
type EndpointsMap map[ServicePortName][]Endpoint

// Update updates em based on the changes in ect, returns information about the diff since
// the last Update, triggers processEndpointsMapChange on every change, and clears the
// changes map.
func (em EndpointsMap) Update(ect *EndpointsChangeTracker) UpdateEndpointsMapResult {
	result := UpdateEndpointsMapResult{
		UpdatedServices:        sets.New[types.NamespacedName](),
		LastChangeTriggerTimes: make(map[types.NamespacedName][]time.Time),
	}
	if ect == nil {
		return result
	}

	changes := ect.checkoutChanges()
	for nn, change := range changes {
		if ect.processEndpointsMapChange != nil {
			ect.processEndpointsMapChange(change.previous, change.current)
		}
		result.UpdatedServices.Insert(nn)

		em.unmerge(change.previous)
		em.merge(change.current)

		// result.ConntrackCleanupRequired should be true if any one of the UDP
		// ServicePort changed endpoint. Once true, we don't update the value.
		if result.ConntrackCleanupRequired {
			continue
		}
		// Check if the changed service had any UDP ServicePort
		for svcPort := range change.previous {
			if svcPort.NamespacedName == nn && svcPort.Protocol == v1.ProtocolUDP {
				result.ConntrackCleanupRequired = true
				break
			}
		}
		// Check if the changed service has any UDP ServicePort
		for svcPort := range change.current {
			if svcPort.NamespacedName == nn && svcPort.Protocol == v1.ProtocolUDP {
				result.ConntrackCleanupRequired = true
				break
			}
		}
	}
	ect.checkoutTriggerTimes(&result.LastChangeTriggerTimes)

	return result
}

// Merge ensures that the current EndpointsMap contains all <service, endpoints> pairs from the EndpointsMap passed in.
func (em EndpointsMap) merge(other EndpointsMap) {
	for svcPortName := range other {
		em[svcPortName] = other[svcPortName]
	}
}

// Unmerge removes the <service, endpoints> pairs from the current EndpointsMap which are contained in the EndpointsMap passed in.
func (em EndpointsMap) unmerge(other EndpointsMap) {
	for svcPortName := range other {
		delete(em, svcPortName)
	}
}

// getLocalEndpointIPs returns endpoints IPs if given endpoint is local - local means the endpoint is running in same host as kube-proxy.
func (em EndpointsMap) getLocalReadyEndpointIPs() map[types.NamespacedName]sets.Set[string] {
	localIPs := make(map[types.NamespacedName]sets.Set[string])
	for svcPortName, epList := range em {
		for _, ep := range epList {
			// Only add ready endpoints for health checking. Terminating endpoints may still serve traffic
			// but the health check signal should fail if there are only terminating endpoints on a node.
			if !ep.IsReady() {
				continue
			}

			if ep.IsLocal() {
				nsn := svcPortName.NamespacedName
				if localIPs[nsn] == nil {
					localIPs[nsn] = sets.New[string]()
				}
				localIPs[nsn].Insert(ep.IP())
			}
		}
	}
	return localIPs
}

// LocalReadyEndpoints returns a map of Service names to the number of local ready
// endpoints for that service.
func (em EndpointsMap) LocalReadyEndpoints() map[types.NamespacedName]int {
	// TODO: If this will appear to be computationally expensive, consider
	// computing this incrementally similarly to endpointsMap.

	// (Note that we need to call getLocalEndpointIPs first to squash the data by IP,
	// because the EndpointsMap is sorted by IP+port, not just IP, and we want to
	// consider a Service pointing to 10.0.0.1:80 and 10.0.0.1:443 to have 1 endpoint,
	// not 2.)

	eps := make(map[types.NamespacedName]int)
	localIPs := em.getLocalReadyEndpointIPs()
	for nsn, ips := range localIPs {
		eps[nsn] = len(ips)
	}
	return eps
}
