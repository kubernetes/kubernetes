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
	"sync"

	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
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
	nodeName         string
}

// endpointSliceTracker keeps track of EndpointSlices as they have been applied
// by a proxier along with any pending EndpointSlices that have been updated
// in this cache but not yet applied by a proxier.
type endpointSliceTracker struct {
	applied endpointSliceDataByName
	pending endpointSliceDataByName
}

// endpointSliceDataByName groups endpointSliceData by the names of the
// corresponding EndpointSlices.
type endpointSliceDataByName map[string]*endpointSliceData

// endpointSliceData contains information about a single EndpointSlice update or removal.
type endpointSliceData struct {
	endpointSlice *discovery.EndpointSlice
	remove        bool
}

// NewEndpointSliceCache initializes an EndpointSliceCache.
func NewEndpointSliceCache(nodeName string, makeEndpointInfo makeEndpointFunc) *EndpointSliceCache {
	if makeEndpointInfo == nil {
		makeEndpointInfo = standardEndpointInfo
	}
	return &EndpointSliceCache{
		trackerByServiceMap: map[types.NamespacedName]*endpointSliceTracker{},
		nodeName:            nodeName,
		makeEndpointInfo:    makeEndpointInfo,
	}
}

// newEndpointSliceTracker initializes an endpointSliceTracker.
func newEndpointSliceTracker() *endpointSliceTracker {
	return &endpointSliceTracker{
		applied: endpointSliceDataByName{},
		pending: endpointSliceDataByName{},
	}
}

// standardEndpointInfo is the default makeEndpointFunc.
func standardEndpointInfo(ep *BaseEndpointInfo, _ *ServicePortName) Endpoint {
	return ep
}

// updatePending updates a pending slice in the cache.
func (cache *EndpointSliceCache) updatePending(endpointSlice *discovery.EndpointSlice, remove bool) bool {
	serviceKey, sliceKey, err := endpointSliceCacheKeys(endpointSlice)
	if err != nil {
		klog.ErrorS(err, "Error getting endpoint slice cache keys")
		return false
	}

	esData := &endpointSliceData{endpointSlice, remove}

	cache.lock.Lock()
	defer cache.lock.Unlock()

	if _, ok := cache.trackerByServiceMap[serviceKey]; !ok {
		cache.trackerByServiceMap[serviceKey] = newEndpointSliceTracker()
	}

	changed := cache.esDataChanged(serviceKey, sliceKey, esData)

	if changed {
		cache.trackerByServiceMap[serviceKey].pending[sliceKey] = esData
	}

	return changed
}

// checkoutChanges returns a map of all endpointsChanges that are
// pending and then marks them as applied.
func (cache *EndpointSliceCache) checkoutChanges() map[types.NamespacedName]*endpointsChange {
	changes := make(map[types.NamespacedName]*endpointsChange)

	cache.lock.Lock()
	defer cache.lock.Unlock()

	for serviceNN, esTracker := range cache.trackerByServiceMap {
		if len(esTracker.pending) == 0 {
			continue
		}

		change := &endpointsChange{}

		change.previous = cache.getEndpointsMap(serviceNN, esTracker.applied)

		for name, sliceData := range esTracker.pending {
			if sliceData.remove {
				delete(esTracker.applied, name)
			} else {
				esTracker.applied[name] = sliceData
			}

			delete(esTracker.pending, name)
			if len(esTracker.applied) == 0 && len(esTracker.pending) == 0 {
				delete(cache.trackerByServiceMap, serviceNN)
			}
		}

		change.current = cache.getEndpointsMap(serviceNN, esTracker.applied)
		changes[serviceNN] = change
	}

	return changes
}

// spToEndpointMap stores groups Endpoint objects by ServicePortName and
// endpoint string (returned by Endpoint.String()).
type spToEndpointMap map[ServicePortName]map[string]Endpoint

// getEndpointsMap computes an EndpointsMap for a given set of EndpointSlices.
func (cache *EndpointSliceCache) getEndpointsMap(serviceNN types.NamespacedName, sliceDataByName endpointSliceDataByName) EndpointsMap {
	endpointInfoBySP := cache.endpointInfoByServicePort(serviceNN, sliceDataByName)
	return endpointsMapFromEndpointInfo(endpointInfoBySP)
}

// endpointInfoByServicePort groups endpoint info by service port name and address.
func (cache *EndpointSliceCache) endpointInfoByServicePort(serviceNN types.NamespacedName, sliceDataByName endpointSliceDataByName) spToEndpointMap {
	endpointInfoBySP := spToEndpointMap{}

	for _, sliceData := range sliceDataByName {
		for _, port := range sliceData.endpointSlice.Ports {
			if port.Name == nil {
				klog.ErrorS(nil, "Ignoring port with nil name", "portName", port.Name)
				continue
			}
			// TODO: handle nil ports to mean "all"
			if port.Port == nil || *port.Port == int32(0) {
				klog.ErrorS(nil, "Ignoring invalid endpoint port", "portName", *port.Name)
				continue
			}

			svcPortName := ServicePortName{
				NamespacedName: serviceNN,
				Port:           *port.Name,
				Protocol:       *port.Protocol,
			}

			endpointInfoBySP[svcPortName] = cache.addEndpoints(&svcPortName, int(*port.Port), endpointInfoBySP[svcPortName], sliceData.endpointSlice.Endpoints)
		}
	}

	return endpointInfoBySP
}

// addEndpoints adds an Endpoint for each unique endpoint.
func (cache *EndpointSliceCache) addEndpoints(svcPortName *ServicePortName, portNum int, endpointSet map[string]Endpoint, endpoints []discovery.Endpoint) map[string]Endpoint {
	if endpointSet == nil {
		endpointSet = map[string]Endpoint{}
	}

	// iterate through endpoints to add them to endpointSet.
	for _, endpoint := range endpoints {
		if len(endpoint.Addresses) == 0 {
			klog.ErrorS(nil, "Ignoring invalid endpoint port with empty address", "endpoint", endpoint)
			continue
		}

		isLocal := endpoint.NodeName != nil && cache.isLocal(*endpoint.NodeName)

		ready := endpoint.Conditions.Ready == nil || *endpoint.Conditions.Ready
		serving := endpoint.Conditions.Serving == nil || *endpoint.Conditions.Serving
		terminating := endpoint.Conditions.Terminating != nil && *endpoint.Conditions.Terminating

		var zoneHints, nodeHints sets.Set[string]
		if endpoint.Hints != nil {
			if len(endpoint.Hints.ForZones) > 0 {
				zoneHints = sets.New[string]()
				for _, zone := range endpoint.Hints.ForZones {
					zoneHints.Insert(zone.Name)
				}
			}
			if len(endpoint.Hints.ForNodes) > 0 && utilfeature.DefaultFeatureGate.Enabled(features.PreferSameTrafficDistribution) {
				nodeHints = sets.New[string]()
				for _, node := range endpoint.Hints.ForNodes {
					nodeHints.Insert(node.Name)
				}
			}
		}

		endpointIP := utilnet.ParseIPSloppy(endpoint.Addresses[0]).String()
		endpointInfo := newBaseEndpointInfo(endpointIP, portNum, isLocal,
			ready, serving, terminating, zoneHints, nodeHints)

		// If an Endpoint gets moved from one slice to another, we may temporarily
		// see it in both slices. Ideally we want to prefer the Endpoint from the
		// more-recently-updated EndpointSlice, since it may have newer
		// conditions. But we can't easily figure that out, and the situation will
		// resolve itself once we receive the second EndpointSlice update anyway.
		//
		// On the other hand, there maybe also be two *different* Endpoints (i.e.,
		// with different targetRefs) that point to the same IP, if the pod
		// network reuses the IP from a terminating pod before the Pod object is
		// fully deleted. In this case we want to prefer the running pod over the
		// terminating one. (If there are multiple non-terminating pods with the
		// same podIP, then the result is undefined.)
		if _, exists := endpointSet[endpointInfo.String()]; !exists || !terminating {
			endpointSet[endpointInfo.String()] = cache.makeEndpointInfo(endpointInfo, svcPortName)
		}
	}

	return endpointSet
}

func (cache *EndpointSliceCache) isLocal(nodeName string) bool {
	return len(cache.nodeName) > 0 && nodeName == cache.nodeName
}

// esDataChanged returns true if the esData parameter should be set as a new
// pending value in the cache.
func (cache *EndpointSliceCache) esDataChanged(serviceKey types.NamespacedName, sliceKey string, esData *endpointSliceData) bool {
	if _, ok := cache.trackerByServiceMap[serviceKey]; ok {
		appliedData, appliedOk := cache.trackerByServiceMap[serviceKey].applied[sliceKey]
		pendingData, pendingOk := cache.trackerByServiceMap[serviceKey].pending[sliceKey]

		// If there's already a pending value, return whether or not this would
		// change that.
		if pendingOk {
			return !reflect.DeepEqual(esData, pendingData)
		}

		// If there's already an applied value, return whether or not this would
		// change that.
		if appliedOk {
			return !reflect.DeepEqual(esData, appliedData)
		}
	}

	// If this is marked for removal and does not exist in the cache, no changes
	// are necessary.
	if esData.remove {
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
	for svcPortName, endpointSet := range endpointInfoBySP {
		if len(endpointSet) > 0 {
			endpointsMap[svcPortName] = []Endpoint{}
			for _, endpointInfo := range endpointSet {
				endpointsMap[svcPortName] = append(endpointsMap[svcPortName], endpointInfo)

			}
			// Ensure endpoints are always returned in the same order to simplify diffing.
			sort.Sort(byEndpoint(endpointsMap[svcPortName]))

			klog.V(3).InfoS("Setting endpoints for service port name", "portName", svcPortName, "endpoints", formatEndpointsList(endpointsMap[svcPortName]))
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
		err = fmt.Errorf("no %s label set on endpoint slice: %s", discovery.LabelServiceName, endpointSlice.Name)
	} else if endpointSlice.Namespace == "" || endpointSlice.Name == "" {
		err = fmt.Errorf("expected EndpointSlice name and namespace to be set: %v", endpointSlice)
	}
	return types.NamespacedName{Namespace: endpointSlice.Namespace, Name: serviceName}, endpointSlice.Name, err
}

// byEndpoint helps sort endpoints by endpoint string.
type byEndpoint []Endpoint

func (e byEndpoint) Len() int {
	return len(e)
}
func (e byEndpoint) Swap(i, j int) {
	e[i], e[j] = e[j], e[i]
}
func (e byEndpoint) Less(i, j int) bool {
	return e[i].String() < e[j].String()
}
