/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"

	"k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	utilnet "k8s.io/utils/net"
)

// EndpointSliceCache is used as a cache of EndpointSlice information.
type EndpointSliceCache struct {
	// lock protects trackerByServiceMap.
	lock sync.Mutex

	// trackerByServiceMap is the basis of this cache. It contains endpoint
	// slice trackers grouped by service name and endpoint slice name. The first
	// key represents a namespaced service name while the second key represents
	// an endpoint slice name. Since endpoints can move between slices, we
	// require slice specific caching to prevent endpoints being removed from
	// the cache when they may have just moved to a different slice.
	trackerByServiceMap map[types.NamespacedName]*endpointSliceTracker

	makeEndpointInfo makeEndpointFunc
	hostname         string
	ipFamily         v1.IPFamily
	recorder         record.EventRecorder
}

// endpointSliceTracker keeps track of EndpointSlices as they have been applied
// by a proxier along with any pending EndpointSlices that have been updated
// in this cache but not yet applied by a proxier.
type endpointSliceTracker struct {
	applied endpointSliceInfoByName
	pending endpointSliceInfoByName
}

// endpointSliceInfoByName groups endpointSliceInfo by the names of the
// corresponding EndpointSlices.
type endpointSliceInfoByName map[string]*endpointSliceInfo

// endpointSliceInfo contains just the attributes kube-proxy cares about.
// Used for caching. Intentionally small to limit memory util.
type endpointSliceInfo struct {
	Ports     []discovery.EndpointPort
	Endpoints []*endpointInfo
	Remove    bool
}

// endpointInfo contains just the attributes kube-proxy cares about.
// Used for caching. Intentionally small to limit memory util.
// Addresses and Topology are copied from EndpointSlice Endpoints.
type endpointInfo struct {
	Addresses []string
	Topology  map[string]string
}

// spToEndpointMap stores groups Endpoint objects by ServicePortName and
// IP address.
type spToEndpointMap map[ServicePortName]map[string]Endpoint

// NewEndpointSliceCache initializes an EndpointSliceCache.
func NewEndpointSliceCache(hostname string, ipFamily v1.IPFamily, recorder record.EventRecorder, makeEndpointInfo makeEndpointFunc) *EndpointSliceCache {
	if makeEndpointInfo == nil {
		makeEndpointInfo = standardEndpointInfo
	}
	return &EndpointSliceCache{
		trackerByServiceMap: map[types.NamespacedName]*endpointSliceTracker{},
		hostname:            hostname,
		ipFamily:            ipFamily,
		makeEndpointInfo:    makeEndpointInfo,
		recorder:            recorder,
	}
}

// newEndpointSliceTracker initializes an endpointSliceTracker.
func newEndpointSliceTracker() *endpointSliceTracker {
	return &endpointSliceTracker{
		applied: endpointSliceInfoByName{},
		pending: endpointSliceInfoByName{},
	}
}

// newEndpointSliceInfo generates endpointSliceInfo from an EndpointSlice.
func newEndpointSliceInfo(endpointSlice *discovery.EndpointSlice, remove bool) *endpointSliceInfo {
	esInfo := &endpointSliceInfo{
		Ports:     make([]discovery.EndpointPort, len(endpointSlice.Ports)),
		Endpoints: []*endpointInfo{},
		Remove:    remove,
	}

	// copy here to avoid mutating shared EndpointSlice object.
	copy(esInfo.Ports, endpointSlice.Ports)
	sort.Sort(byPort(esInfo.Ports))

	if !remove {
		for _, endpoint := range endpointSlice.Endpoints {
			if endpoint.Conditions.Ready == nil || *endpoint.Conditions.Ready {
				esInfo.Endpoints = append(esInfo.Endpoints, &endpointInfo{
					Addresses: endpoint.Addresses,
					Topology:  endpoint.Topology,
				})
			}
		}

		sort.Sort(byAddress(esInfo.Endpoints))
	}

	return esInfo
}

// standardEndpointInfo is the default makeEndpointFunc.
func standardEndpointInfo(ep *BaseEndpointInfo) Endpoint {
	return ep
}

// updatePending updates a pending slice in the cache.
func (cache *EndpointSliceCache) updatePending(endpointSlice *discovery.EndpointSlice, remove bool) bool {
	serviceKey, sliceKey, err := endpointSliceCacheKeys(endpointSlice)
	if err != nil {
		klog.Warningf("Error getting endpoint slice cache keys: %v", err)
		return false
	}

	esInfo := newEndpointSliceInfo(endpointSlice, remove)

	cache.lock.Lock()
	defer cache.lock.Unlock()

	if _, ok := cache.trackerByServiceMap[serviceKey]; !ok {
		cache.trackerByServiceMap[serviceKey] = newEndpointSliceTracker()
	}

	changed := cache.esInfoChanged(serviceKey, sliceKey, esInfo)

	if changed {
		cache.trackerByServiceMap[serviceKey].pending[sliceKey] = esInfo
	}

	return changed
}

// checkoutChanges returns a list of all endpointsChanges that are
// pending and then marks them as applied.
func (cache *EndpointSliceCache) checkoutChanges() []*endpointsChange {
	changes := []*endpointsChange{}

	cache.lock.Lock()
	defer cache.lock.Unlock()

	for serviceNN, esTracker := range cache.trackerByServiceMap {
		if len(esTracker.pending) == 0 {
			continue
		}

		change := &endpointsChange{}

		change.previous = cache.getEndpointsMap(serviceNN, esTracker.applied)

		for name, sliceInfo := range esTracker.pending {
			if sliceInfo.Remove {
				delete(esTracker.applied, name)
			} else {
				esTracker.applied[name] = sliceInfo
			}

			delete(esTracker.pending, name)
		}

		change.current = cache.getEndpointsMap(serviceNN, esTracker.applied)
		changes = append(changes, change)
	}

	return changes
}

// getEndpointsMap computes an EndpointsMap for a given set of EndpointSlices.
func (cache *EndpointSliceCache) getEndpointsMap(serviceNN types.NamespacedName, sliceInfoByName endpointSliceInfoByName) EndpointsMap {
	endpointInfoBySP := cache.endpointInfoByServicePort(serviceNN, sliceInfoByName)
	return endpointsMapFromEndpointInfo(endpointInfoBySP)
}

// endpointInfoByServicePort groups endpoint info by service port name and address.
func (cache *EndpointSliceCache) endpointInfoByServicePort(serviceNN types.NamespacedName, sliceInfoByName endpointSliceInfoByName) spToEndpointMap {
	endpointInfoBySP := spToEndpointMap{}

	for _, sliceInfo := range sliceInfoByName {
		for _, port := range sliceInfo.Ports {
			if port.Name == nil {
				klog.Warningf("ignoring port with nil name %v", port)
				continue
			}
			// TODO: handle nil ports to mean "all"
			if port.Port == nil || *port.Port == int32(0) {
				klog.Warningf("ignoring invalid endpoint port %s", *port.Name)
				continue
			}

			svcPortName := ServicePortName{
				NamespacedName: serviceNN,
				Port:           *port.Name,
				Protocol:       *port.Protocol,
			}

			endpointInfoBySP[svcPortName] = cache.addEndpointsByIP(serviceNN, int(*port.Port), endpointInfoBySP[svcPortName], sliceInfo.Endpoints)
		}
	}

	return endpointInfoBySP
}

// addEndpointsByIP adds endpointInfo for each IP.
func (cache *EndpointSliceCache) addEndpointsByIP(serviceNN types.NamespacedName, portNum int, endpointsByIP map[string]Endpoint, endpoints []*endpointInfo) map[string]Endpoint {
	if endpointsByIP == nil {
		endpointsByIP = map[string]Endpoint{}
	}

	// iterate through endpoints to add them to endpointsByIP.
	for _, endpoint := range endpoints {
		if len(endpoint.Addresses) == 0 {
			klog.Warningf("ignoring invalid endpoint port %s with empty addresses", endpoint)
			continue
		}

		// Filter out the incorrect IP version case. Any endpoint port that
		// contains incorrect IP version will be ignored.
		if (cache.ipFamily == v1.IPv6Protocol) != utilnet.IsIPv6String(endpoint.Addresses[0]) {
			// Emit event on the corresponding service which had a different IP
			// version than the endpoint.
			utilproxy.LogAndEmitIncorrectIPVersionEvent(cache.recorder, "endpointslice", endpoint.Addresses[0], serviceNN.Namespace, serviceNN.Name, "")
			continue
		}

		isLocal := cache.isLocal(endpoint.Topology[v1.LabelHostname])
		endpointInfo := newBaseEndpointInfo(endpoint.Addresses[0], portNum, isLocal, endpoint.Topology)

		// This logic ensures we're deduping potential overlapping endpoints
		// isLocal should not vary between matching IPs, but if it does, we
		// favor a true value here if it exists.
		if _, exists := endpointsByIP[endpointInfo.IP()]; !exists || isLocal {
			endpointsByIP[endpointInfo.IP()] = cache.makeEndpointInfo(endpointInfo)
		}
	}

	return endpointsByIP
}

func (cache *EndpointSliceCache) isLocal(hostname string) bool {
	return len(cache.hostname) > 0 && hostname == cache.hostname
}

// esInfoChanged returns true if the esInfo parameter should be set as a new
// pending value in the cache.
func (cache *EndpointSliceCache) esInfoChanged(serviceKey types.NamespacedName, sliceKey string, esInfo *endpointSliceInfo) bool {
	if _, ok := cache.trackerByServiceMap[serviceKey]; ok {
		appliedInfo, appliedOk := cache.trackerByServiceMap[serviceKey].applied[sliceKey]
		pendingInfo, pendingOk := cache.trackerByServiceMap[serviceKey].pending[sliceKey]

		// If there's already a pending value, return whether or not this would
		// change that.
		if pendingOk {
			return !reflect.DeepEqual(esInfo, pendingInfo)
		}

		// If there's already an applied value, return whether or not this would
		// change that.
		if appliedOk {
			return !reflect.DeepEqual(esInfo, appliedInfo)
		}
	}

	// If this is marked for removal and does not exist in the cache, no changes
	// are necessary.
	if esInfo.Remove {
		return false
	}

	// If not in the cache, and not marked for removal, it should be added.
	return true
}

// endpointsMapFromEndpointInfo computes an endpointsMap from endpointInfo that
// has been grouped by service port and IP.
func endpointsMapFromEndpointInfo(endpointInfoBySP map[ServicePortName]map[string]Endpoint) EndpointsMap {
	endpointsMap := EndpointsMap{}

	// transform endpointInfoByServicePort into an endpointsMap with sorted IPs.
	for svcPortName, endpointInfoByIP := range endpointInfoBySP {
		if len(endpointInfoByIP) > 0 {
			endpointsMap[svcPortName] = []Endpoint{}
			for _, endpointInfo := range endpointInfoByIP {
				endpointsMap[svcPortName] = append(endpointsMap[svcPortName], endpointInfo)

			}
			// Ensure IPs are always returned in the same order to simplify diffing.
			sort.Sort(byIP(endpointsMap[svcPortName]))

			klog.V(3).Infof("Setting endpoints for %q to %+v", svcPortName, formatEndpointsList(endpointsMap[svcPortName]))
		}
	}

	return endpointsMap
}

// formatEndpointsList returns a string list converted from an endpoints list.
func formatEndpointsList(endpoints []Endpoint) []string {
	var formattedList []string
	for _, ep := range endpoints {
		formattedList = append(formattedList, ep.String())
	}
	return formattedList
}

// endpointSliceCacheKeys returns cache keys used for a given EndpointSlice.
func endpointSliceCacheKeys(endpointSlice *discovery.EndpointSlice) (types.NamespacedName, string, error) {
	var err error
	serviceName, ok := endpointSlice.Labels[discovery.LabelServiceName]
	if !ok || serviceName == "" {
		err = fmt.Errorf("No %s label set on endpoint slice: %s", discovery.LabelServiceName, endpointSlice.Name)
	} else if endpointSlice.Namespace == "" || endpointSlice.Name == "" {
		err = fmt.Errorf("Expected EndpointSlice name and namespace to be set: %v", endpointSlice)
	}
	return types.NamespacedName{Namespace: endpointSlice.Namespace, Name: serviceName}, endpointSlice.Name, err
}

// byAddress helps sort endpointInfo
type byAddress []*endpointInfo

func (e byAddress) Len() int {
	return len(e)
}
func (e byAddress) Swap(i, j int) {
	e[i], e[j] = e[j], e[i]
}
func (e byAddress) Less(i, j int) bool {
	return strings.Join(e[i].Addresses, ",") < strings.Join(e[j].Addresses, ",")
}

// byIP helps sort endpoints by IP
type byIP []Endpoint

func (e byIP) Len() int {
	return len(e)
}
func (e byIP) Swap(i, j int) {
	e[i], e[j] = e[j], e[i]
}
func (e byIP) Less(i, j int) bool {
	return e[i].String() < e[j].String()
}

// byPort helps sort EndpointSlice ports by port number
type byPort []discovery.EndpointPort

func (p byPort) Len() int {
	return len(p)
}
func (p byPort) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}
func (p byPort) Less(i, j int) bool {
	return *p[i].Port < *p[j].Port
}
