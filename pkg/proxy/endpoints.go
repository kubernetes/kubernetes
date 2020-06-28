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
	"reflect"
	"strconv"
	"sync"
	"time"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	utilnet "k8s.io/utils/net"
)

var supportedEndpointSliceAddressTypes = sets.NewString(
	string(discovery.AddressTypeIP), // IP is a deprecated address type
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
	IsLocal  bool
	Topology map[string]string
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

// GetTopology returns the topology information of the endpoint.
func (info *BaseEndpointInfo) GetTopology() map[string]string {
	return info.Topology
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
	return info.String() == other.String() && info.GetIsLocal() == other.GetIsLocal()
}

func newBaseEndpointInfo(IP string, port int, isLocal bool, topology map[string]string) *BaseEndpointInfo {
	return &BaseEndpointInfo{
		Endpoint: net.JoinHostPort(IP, strconv.Itoa(port)),
		IsLocal:  isLocal,
		Topology: topology,
	}
}

type makeEndpointFunc func(info *BaseEndpointInfo) Endpoint

// EndpointChangeTracker carries state about uncommitted changes to an arbitrary number of
// Endpoints, keyed by their namespace and name.
type EndpointChangeTracker struct {
	// lock protects items.
	lock sync.Mutex
	// hostname is the host where kube-proxy is running.
	hostname string
	// items maps a service to is endpointsChange.
	items map[types.NamespacedName]*endpointsChange
	// makeEndpointInfo allows proxier to inject customized information when processing endpoint.
	makeEndpointInfo makeEndpointFunc
	// endpointSliceCache holds a simplified version of endpoint slices.
	endpointSliceCache *EndpointSliceCache
	// isIPv6Mode indicates if change tracker is under IPv6/IPv4 mode. Nil means not applicable.
	isIPv6Mode *bool
	recorder   record.EventRecorder
	// Map from the Endpoints namespaced-name to the times of the triggers that caused the endpoints
	// object to change. Used to calculate the network-programming-latency.
	lastChangeTriggerTimes map[types.NamespacedName][]time.Time
}

// NewEndpointChangeTracker initializes an EndpointsChangeMap
func NewEndpointChangeTracker(hostname string, makeEndpointInfo makeEndpointFunc, isIPv6Mode *bool, recorder record.EventRecorder, endpointSlicesEnabled bool) *EndpointChangeTracker {
	ect := &EndpointChangeTracker{
		hostname:               hostname,
		items:                  make(map[types.NamespacedName]*endpointsChange),
		makeEndpointInfo:       makeEndpointInfo,
		isIPv6Mode:             isIPv6Mode,
		recorder:               recorder,
		lastChangeTriggerTimes: make(map[types.NamespacedName][]time.Time),
	}
	if endpointSlicesEnabled {
		ect.endpointSliceCache = NewEndpointSliceCache(hostname, isIPv6Mode, recorder, makeEndpointInfo)
	}
	return ect
}

// Update updates given service's endpoints change map based on the <previous, current> endpoints pair.  It returns true
// if items changed, otherwise return false.  Update can be used to add/update/delete items of EndpointsChangeMap.  For example,
// Add item
//   - pass <nil, endpoints> as the <previous, current> pair.
// Update item
//   - pass <oldEndpoints, endpoints> as the <previous, current> pair.
// Delete item
//   - pass <endpoints, nil> as the <previous, current> pair.
func (ect *EndpointChangeTracker) Update(previous, current *v1.Endpoints) bool {
	endpoints := current
	if endpoints == nil {
		endpoints = previous
	}
	// previous == nil && current == nil is unexpected, we should return false directly.
	if endpoints == nil {
		return false
	}
	metrics.EndpointChangesTotal.Inc()
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}

	ect.lock.Lock()
	defer ect.lock.Unlock()

	change, exists := ect.items[namespacedName]
	if !exists {
		change = &endpointsChange{}
		change.previous = ect.endpointsToEndpointsMap(previous)
		ect.items[namespacedName] = change
	}

	if t := getLastChangeTriggerTime(endpoints.Annotations); !t.IsZero() {
		ect.lastChangeTriggerTimes[namespacedName] = append(ect.lastChangeTriggerTimes[namespacedName], t)
	}

	change.current = ect.endpointsToEndpointsMap(current)
	// if change.previous equal to change.current, it means no change
	if reflect.DeepEqual(change.previous, change.current) {
		delete(ect.items, namespacedName)
		// Reset the lastChangeTriggerTimes for the Endpoints object. Given that the network programming
		// SLI is defined as the duration between a time of an event and a time when the network was
		// programmed to incorporate that event, if there are events that happened between two
		// consecutive syncs and that canceled each other out, e.g. pod A added -> pod A deleted,
		// there will be no network programming for them and thus no network programming latency metric
		// should be exported.
		delete(ect.lastChangeTriggerTimes, namespacedName)
	} else {
		for spn, eps := range change.current {
			klog.V(2).Infof("Service port %s updated: %d endpoints", spn, len(eps))
		}
	}

	metrics.EndpointChangesPending.Set(float64(len(ect.items)))
	return len(ect.items) > 0
}

// EndpointSliceUpdate updates given service's endpoints change map based on the <previous, current> endpoints pair.
// It returns true if items changed, otherwise return false. Will add/update/delete items of EndpointsChangeMap.
// If removeSlice is true, slice will be removed, otherwise it will be added or updated.
func (ect *EndpointChangeTracker) EndpointSliceUpdate(endpointSlice *discovery.EndpointSlice, removeSlice bool) bool {
	if !supportedEndpointSliceAddressTypes.Has(string(endpointSlice.AddressType)) {
		klog.V(4).Infof("EndpointSlice address type not supported by kube-proxy: %s", endpointSlice.AddressType)
		return false
	}

	// This should never happen
	if endpointSlice == nil {
		klog.Error("Nil endpointSlice passed to EndpointSliceUpdate")
		return false
	}

	namespacedName, _, err := endpointSliceCacheKeys(endpointSlice)
	if err != nil {
		klog.Warningf("Error getting endpoint slice cache keys: %v", err)
		return false
	}

	metrics.EndpointChangesTotal.Inc()

	ect.lock.Lock()
	defer ect.lock.Unlock()

	changeNeeded := ect.endpointSliceCache.updatePending(endpointSlice, removeSlice)

	if changeNeeded {
		metrics.EndpointChangesPending.Inc()
		if t := getLastChangeTriggerTime(endpointSlice.Annotations); !t.IsZero() {
			ect.lastChangeTriggerTimes[namespacedName] =
				append(ect.lastChangeTriggerTimes[namespacedName], t)
		}
	}

	return changeNeeded
}

// checkoutChanges returns a list of pending endpointsChanges and marks them as
// applied.
func (ect *EndpointChangeTracker) checkoutChanges() []*endpointsChange {
	ect.lock.Lock()
	defer ect.lock.Unlock()

	metrics.EndpointChangesPending.Set(0)

	if ect.endpointSliceCache != nil {
		return ect.endpointSliceCache.checkoutChanges()
	}

	changes := []*endpointsChange{}
	for _, change := range ect.items {
		changes = append(changes, change)
	}
	ect.items = make(map[types.NamespacedName]*endpointsChange)
	return changes
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
		klog.Warningf("Error while parsing EndpointsLastChangeTriggerTimeAnnotation: '%s'. Error is %v",
			annotations[v1.EndpointsLastChangeTriggerTime], err)
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
	localIPs := em.getLocalEndpointIPs()
	for nsn, ips := range localIPs {
		result.HCEndpointsLocalIPSize[nsn] = len(ips)
	}

	return result
}

// EndpointsMap maps a service name to a list of all its Endpoints.
type EndpointsMap map[ServicePortName][]Endpoint

// endpointsToEndpointsMap translates single Endpoints object to EndpointsMap.
// This function is used for incremental updated of endpointsMap.
//
// NOTE: endpoints object should NOT be modified.
func (ect *EndpointChangeTracker) endpointsToEndpointsMap(endpoints *v1.Endpoints) EndpointsMap {
	if endpoints == nil {
		return nil
	}

	endpointsMap := make(EndpointsMap)
	// We need to build a map of portname -> all ip:ports for that portname.
	// Explode Endpoints.Subsets[*] into this structure.
	for i := range endpoints.Subsets {
		ss := &endpoints.Subsets[i]
		for i := range ss.Ports {
			port := &ss.Ports[i]
			if port.Port == 0 {
				klog.Warningf("ignoring invalid endpoint port %s", port.Name)
				continue
			}
			svcPortName := ServicePortName{
				NamespacedName: types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name},
				Port:           port.Name,
				Protocol:       port.Protocol,
			}
			for i := range ss.Addresses {
				addr := &ss.Addresses[i]
				if addr.IP == "" {
					klog.Warningf("ignoring invalid endpoint port %s with empty host", port.Name)
					continue
				}
				// Filter out the incorrect IP version case.
				// Any endpoint port that contains incorrect IP version will be ignored.
				if ect.isIPv6Mode != nil && utilnet.IsIPv6String(addr.IP) != *ect.isIPv6Mode {
					// Emit event on the corresponding service which had a different
					// IP version than the endpoint.
					utilproxy.LogAndEmitIncorrectIPVersionEvent(ect.recorder, "endpoints", addr.IP, endpoints.Namespace, endpoints.Name, "")
					continue
				}
				isLocal := addr.NodeName != nil && *addr.NodeName == ect.hostname
				baseEndpointInfo := newBaseEndpointInfo(addr.IP, int(port.Port), isLocal, nil)
				if ect.makeEndpointInfo != nil {
					endpointsMap[svcPortName] = append(endpointsMap[svcPortName], ect.makeEndpointInfo(baseEndpointInfo))
				} else {
					endpointsMap[svcPortName] = append(endpointsMap[svcPortName], baseEndpointInfo)
				}
			}

			klog.V(3).Infof("Setting endpoints for %q to %+v", svcPortName, formatEndpointsList(endpointsMap[svcPortName]))
		}
	}
	return endpointsMap
}

// apply the changes to EndpointsMap and updates stale endpoints and service-endpoints pair. The `staleEndpoints` argument
// is passed in to store the stale udp endpoints and `staleServiceNames` argument is passed in to store the stale udp service.
// The changes map is cleared after applying them.
// In addition it returns (via argument) and resets the lastChangeTriggerTimes for all endpoints
// that were changed and will result in syncing the proxy rules.
func (em EndpointsMap) apply(ect *EndpointChangeTracker, staleEndpoints *[]ServiceEndpoint,
	staleServiceNames *[]ServicePortName, lastChangeTriggerTimes *map[types.NamespacedName][]time.Time) {
	if ect == nil {
		return
	}

	changes := ect.checkoutChanges()
	for _, change := range changes {
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
func (em EndpointsMap) getLocalEndpointIPs() map[types.NamespacedName]sets.String {
	localIPs := make(map[types.NamespacedName]sets.String)
	for svcPortName, epList := range em {
		for _, ep := range epList {
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
	for svcPortName, epList := range oldEndpointsMap {
		if svcPortName.Protocol != v1.ProtocolUDP {
			continue
		}

		for _, ep := range epList {
			stale := true
			for i := range newEndpointsMap[svcPortName] {
				if newEndpointsMap[svcPortName][i].Equal(ep) {
					stale = false
					break
				}
			}
			if stale {
				klog.V(4).Infof("Stale endpoint %v -> %v", svcPortName, ep.String())
				*staleEndpoints = append(*staleEndpoints, ServiceEndpoint{Endpoint: ep.String(), ServicePortName: svcPortName})
			}
		}
	}

	for svcPortName, epList := range newEndpointsMap {
		if svcPortName.Protocol != v1.ProtocolUDP {
			continue
		}

		// For udp service, if its backend changes from 0 to non-0. There may exist a conntrack entry that could blackhole traffic to the service.
		if len(epList) > 0 && len(oldEndpointsMap[svcPortName]) == 0 {
			*staleServiceNames = append(*staleServiceNames, svcPortName)
		}
	}
}
