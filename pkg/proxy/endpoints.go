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
	"net"
	"strconv"
	"sync"
	"time"

	"k8s.io/client-go/tools/events"
	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
)

var supportedEndpointSliceAddressTypes = sets.NewString(
	string(discovery.AddressTypeIPv4),
	string(discovery.AddressTypeIPv6),
)

// BaseEndpointInfo contains base information that defines an endpoint.
// This could be used directly by proxier while processing endpoints,
// or can be used for constructing a more specific EndpointInfo struct
// defined by the proxier if needed.
type BaseEndpointInfo struct {
	Endpoint string // TODO: should be an endpointString type
	// IsLocal indicates whether the endpoint is running in same host as kube-proxy.
	IsLocal bool

	// ZoneHints represent the zone hints for the endpoint. This is based on
	// endpoint.hints.forZones[*].name in the EndpointSlice API.
	ZoneHints sets.String
	// Ready indicates whether this endpoint is ready and NOT terminating.
	// For pods, this is true if a pod has a ready status and a nil deletion timestamp.
	// This is only set when watching EndpointSlices. If using Endpoints, this is always
	// true since only ready endpoints are read from Endpoints.
	// TODO: Ready can be inferred from Serving and Terminating below when enabled by default.
	Ready bool
	// Serving indiciates whether this endpoint is ready regardless of its terminating state.
	// For pods this is true if it has a ready status regardless of its deletion timestamp.
	// This is only set when watching EndpointSlices. If using Endpoints, this is always
	// true since only ready endpoints are read from Endpoints.
	Serving bool
	// Terminating indicates whether this endpoint is terminating.
	// For pods this is true if it has a non-nil deletion timestamp.
	// This is only set when watching EndpointSlices. If using Endpoints, this is always
	// false since terminating endpoints are always excluded from Endpoints.
	Terminating bool

	// NodeName is the name of the node this endpoint belongs to
	NodeName string
	// Zone is the name of the zone this endpoint belongs to
	Zone string
}

var _ Endpoint = &BaseEndpointInfo{}

// String is part of proxy.Endpoint interface.
func (info *BaseEndpointInfo) String() string {
	return info.Endpoint
}

// GetIsLocal is part of proxy.Endpoint interface.
func (info *BaseEndpointInfo) GetIsLocal() bool {
	return info.IsLocal
}

// IsReady returns true if an endpoint is ready and not terminating.
func (info *BaseEndpointInfo) IsReady() bool {
	return info.Ready
}

// IsServing returns true if an endpoint is ready, regardless of if the
// endpoint is terminating.
func (info *BaseEndpointInfo) IsServing() bool {
	return info.Serving
}

// IsTerminating retruns true if an endpoint is terminating. For pods,
// that is any pod with a deletion timestamp.
func (info *BaseEndpointInfo) IsTerminating() bool {
	return info.Terminating
}

// GetZoneHints returns the zone hint for the endpoint.
func (info *BaseEndpointInfo) GetZoneHints() sets.String {
	return info.ZoneHints
}

// IP returns just the IP part of the endpoint, it's a part of proxy.Endpoint interface.
func (info *BaseEndpointInfo) IP() string {
	return utilproxy.IPPart(info.Endpoint)
}

// Port returns just the Port part of the endpoint.
func (info *BaseEndpointInfo) Port() (int, error) {
	return utilproxy.PortPart(info.Endpoint)
}

// Equal is part of proxy.Endpoint interface.
func (info *BaseEndpointInfo) Equal(other Endpoint) bool {
	return info.String() == other.String() &&
		info.GetIsLocal() == other.GetIsLocal() &&
		info.IsReady() == other.IsReady()
}

// GetNodeName returns the NodeName for this endpoint.
func (info *BaseEndpointInfo) GetNodeName() string {
	return info.NodeName
}

// GetZone returns the Zone for this endpoint.
func (info *BaseEndpointInfo) GetZone() string {
	return info.Zone
}

func newBaseEndpointInfo(IP, nodeName, zone string, port int, isLocal bool,
	ready, serving, terminating bool, zoneHints sets.String) *BaseEndpointInfo {
	return &BaseEndpointInfo{
		Endpoint:    net.JoinHostPort(IP, strconv.Itoa(port)),
		IsLocal:     isLocal,
		Ready:       ready,
		Serving:     serving,
		Terminating: terminating,
		ZoneHints:   zoneHints,
		NodeName:    nodeName,
		Zone:        zone,
	}
}

type makeEndpointFunc func(info *BaseEndpointInfo, svcPortName *ServicePortName) Endpoint

// This handler is invoked by the apply function on every change. This function should not modify the
// EndpointsMap's but just use the changes for any Proxier specific cleanup.
type processEndpointsMapChangeFunc func(oldEndpointsMap, newEndpointsMap EndpointsMap)

// EndpointChangeTracker carries state about uncommitted changes to an arbitrary number of
// Endpoints, keyed by their namespace and name.
type EndpointChangeTracker struct {
	// lock protects lastChangeTriggerTimes
	lock sync.Mutex

	processEndpointsMapChange processEndpointsMapChangeFunc
	// endpointSliceCache holds a simplified version of endpoint slices.
	endpointSliceCache *EndpointSliceCache
	// Map from the Endpoints namespaced-name to the times of the triggers that caused the endpoints
	// object to change. Used to calculate the network-programming-latency.
	lastChangeTriggerTimes map[types.NamespacedName][]time.Time
	// record the time when the endpointChangeTracker was created so we can ignore the endpoints
	// that were generated before, because we can't estimate the network-programming-latency on those.
	// This is specially problematic on restarts, because we process all the endpoints that may have been
	// created hours or days before.
	trackerStartTime time.Time
}

// NewEndpointChangeTracker initializes an EndpointsChangeMap
func NewEndpointChangeTracker(hostname string, makeEndpointInfo makeEndpointFunc, ipFamily v1.IPFamily, recorder events.EventRecorder, processEndpointsMapChange processEndpointsMapChangeFunc) *EndpointChangeTracker {
	return &EndpointChangeTracker{
		lastChangeTriggerTimes:    make(map[types.NamespacedName][]time.Time),
		trackerStartTime:          time.Now(),
		processEndpointsMapChange: processEndpointsMapChange,
		endpointSliceCache:        NewEndpointSliceCache(hostname, ipFamily, recorder, makeEndpointInfo),
	}
}

// EndpointSliceUpdate updates given service's endpoints change map based on the <previous, current> endpoints pair.
// It returns true if items changed, otherwise return false. Will add/update/delete items of EndpointsChangeMap.
// If removeSlice is true, slice will be removed, otherwise it will be added or updated.
func (ect *EndpointChangeTracker) EndpointSliceUpdate(endpointSlice *discovery.EndpointSlice, removeSlice bool) bool {
	if !supportedEndpointSliceAddressTypes.Has(string(endpointSlice.AddressType)) {
		klog.V(4).InfoS("EndpointSlice address type not supported by kube-proxy", "addressType", endpointSlice.AddressType)
		return false
	}

	// This should never happen
	if endpointSlice == nil {
		klog.ErrorS(nil, "Nil endpointSlice passed to EndpointSliceUpdate")
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

// PendingChanges returns a set whose keys are the names of the services whose endpoints
// have changed since the last time ect was used to update an EndpointsMap. (You must call
// this _before_ calling em.Update(ect).)
func (ect *EndpointChangeTracker) PendingChanges() sets.String {
	return ect.endpointSliceCache.pendingChanges()
}

// checkoutChanges returns a list of pending endpointsChanges and marks them as
// applied.
func (ect *EndpointChangeTracker) checkoutChanges() []*endpointsChange {
	metrics.EndpointChangesPending.Set(0)

	return ect.endpointSliceCache.checkoutChanges()
}

// checkoutTriggerTimes applies the locally cached trigger times to a map of
// trigger times that have been passed in and empties the local cache.
func (ect *EndpointChangeTracker) checkoutTriggerTimes(lastChangeTriggerTimes *map[types.NamespacedName][]time.Time) {
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

// UpdateEndpointMapResult is the updated results after applying endpoints changes.
type UpdateEndpointMapResult struct {
	// HCEndpointsLocalIPSize maps an endpoints name to the length of its local IPs.
	HCEndpointsLocalIPSize map[types.NamespacedName]int
	// StaleEndpoints identifies if an endpoints service pair is stale.
	StaleEndpoints []ServiceEndpoint
	// StaleServiceNames identifies if a service is stale.
	StaleServiceNames []ServicePortName
	// List of the trigger times for all endpoints objects that changed. It's used to export the
	// network programming latency.
	// NOTE(oxddr): this can be simplified to []time.Time if memory consumption becomes an issue.
	LastChangeTriggerTimes map[types.NamespacedName][]time.Time
}

// Update updates endpointsMap base on the given changes.
func (em EndpointsMap) Update(changes *EndpointChangeTracker) (result UpdateEndpointMapResult) {
	result.StaleEndpoints = make([]ServiceEndpoint, 0)
	result.StaleServiceNames = make([]ServicePortName, 0)
	result.LastChangeTriggerTimes = make(map[types.NamespacedName][]time.Time)

	em.apply(
		changes, &result.StaleEndpoints, &result.StaleServiceNames, &result.LastChangeTriggerTimes)

	// TODO: If this will appear to be computationally expensive, consider
	// computing this incrementally similarly to endpointsMap.
	result.HCEndpointsLocalIPSize = make(map[types.NamespacedName]int)
	localIPs := em.getLocalReadyEndpointIPs()
	for nsn, ips := range localIPs {
		result.HCEndpointsLocalIPSize[nsn] = len(ips)
	}

	return result
}

// EndpointsMap maps a service name to a list of all its Endpoints.
type EndpointsMap map[ServicePortName][]Endpoint

// apply the changes to EndpointsMap and updates stale endpoints and service-endpoints pair. The `staleEndpoints` argument
// is passed in to store the stale udp endpoints and `staleServiceNames` argument is passed in to store the stale udp service.
// The changes map is cleared after applying them.
// In addition it returns (via argument) and resets the lastChangeTriggerTimes for all endpoints
// that were changed and will result in syncing the proxy rules.
// apply triggers processEndpointsMapChange on every change.
func (em EndpointsMap) apply(ect *EndpointChangeTracker, staleEndpoints *[]ServiceEndpoint,
	staleServiceNames *[]ServicePortName, lastChangeTriggerTimes *map[types.NamespacedName][]time.Time) {
	if ect == nil {
		return
	}

	changes := ect.checkoutChanges()
	for _, change := range changes {
		if ect.processEndpointsMapChange != nil {
			ect.processEndpointsMapChange(change.previous, change.current)
		}
		em.unmerge(change.previous)
		em.merge(change.current)
		detectStaleConnections(change.previous, change.current, staleEndpoints, staleServiceNames)
	}
	ect.checkoutTriggerTimes(lastChangeTriggerTimes)
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

// GetLocalEndpointIPs returns endpoints IPs if given endpoint is local - local means the endpoint is running in same host as kube-proxy.
func (em EndpointsMap) getLocalReadyEndpointIPs() map[types.NamespacedName]sets.String {
	localIPs := make(map[types.NamespacedName]sets.String)
	for svcPortName, epList := range em {
		for _, ep := range epList {
			// Only add ready endpoints for health checking. Terminating endpoints may still serve traffic
			// but the health check signal should fail if there are only terminating endpoints on a node.
			if !ep.IsReady() {
				continue
			}

			if ep.GetIsLocal() {
				nsn := svcPortName.NamespacedName
				if localIPs[nsn] == nil {
					localIPs[nsn] = sets.NewString()
				}
				localIPs[nsn].Insert(ep.IP())
			}
		}
	}
	return localIPs
}

// detectStaleConnections modifies <staleEndpoints> and <staleServices> with detected stale connections. <staleServiceNames>
// is used to store stale udp service in order to clear udp conntrack later.
func detectStaleConnections(oldEndpointsMap, newEndpointsMap EndpointsMap, staleEndpoints *[]ServiceEndpoint, staleServiceNames *[]ServicePortName) {
	// Detect stale endpoints: an endpoint can have stale conntrack entries if it was receiving traffic
	// and then goes unready or changes its IP address.
	for svcPortName, epList := range oldEndpointsMap {
		if svcPortName.Protocol != v1.ProtocolUDP {
			continue
		}

		for _, ep := range epList {
			// if the old endpoint wasn't ready is not possible to have stale entries
			// since there was no traffic sent to it.
			if !ep.IsReady() {
				continue
			}
			stale := true
			// Check if the endpoint has changed, including if it went from ready to not ready.
			// If it did change stale entries for the old endpoint has to be cleared.
			for i := range newEndpointsMap[svcPortName] {
				if newEndpointsMap[svcPortName][i].Equal(ep) {
					stale = false
					break
				}
			}
			if stale {
				klog.V(4).InfoS("Stale endpoint", "portName", svcPortName, "endpoint", ep)
				*staleEndpoints = append(*staleEndpoints, ServiceEndpoint{Endpoint: ep.String(), ServicePortName: svcPortName})
			}
		}
	}

	// Detect stale services
	// For udp service, if its backend changes from 0 to non-0 ready endpoints.
	// There may exist a conntrack entry that could blackhole traffic to the service.
	for svcPortName, epList := range newEndpointsMap {
		if svcPortName.Protocol != v1.ProtocolUDP {
			continue
		}

		epReady := 0
		for _, ep := range epList {
			if ep.IsReady() {
				epReady++
			}
		}

		oldEpReady := 0
		for _, ep := range oldEndpointsMap[svcPortName] {
			if ep.IsReady() {
				oldEpReady++
			}
		}

		if epReady > 0 && oldEpReady == 0 {
			*staleServiceNames = append(*staleServiceNames, svcPortName)
		}
	}
}
