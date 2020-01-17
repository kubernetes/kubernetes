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
	"sort"

	"k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1alpha1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	utilnet "k8s.io/utils/net"
)

// EndpointSliceCache is used as a cache of EndpointSlice information.
type EndpointSliceCache struct {
	// sliceByServiceMap is the basis of this cache. It contains endpoint slice
	// info grouped by service name and endpoint slice name. The first key
	// represents a namespaced service name while the second key represents
	// an endpoint slice name. Since endpoints can move between slices, we
	// require slice specific caching to prevent endpoints being removed from
	// the cache when they may have just moved to a different slice.
	sliceByServiceMap map[types.NamespacedName]map[string]*endpointSliceInfo
	makeEndpointInfo  makeEndpointFunc
	hostname          string
	isIPv6Mode        *bool
	recorder          record.EventRecorder
}

// endpointSliceInfo contains just the attributes kube-proxy cares about.
// Used for caching. Intentionally small to limit memory util.
type endpointSliceInfo struct {
	Ports     []discovery.EndpointPort
	Endpoints []*endpointInfo
}

// endpointInfo contains just the attributes kube-proxy cares about.
// Used for caching. Intentionally small to limit memory util.
// Addresses and Topology are copied from EndpointSlice Endpoints.
type endpointInfo struct {
	Addresses []string
	Topology  map[string]string
}

// spToEndpointMap stores groups Endpoint objects by ServicePortName and
// EndpointSlice name.
type spToEndpointMap map[ServicePortName]map[string]Endpoint

// NewEndpointSliceCache initializes an EndpointSliceCache.
func NewEndpointSliceCache(hostname string, isIPv6Mode *bool, recorder record.EventRecorder, makeEndpointInfo makeEndpointFunc) *EndpointSliceCache {
	if makeEndpointInfo == nil {
		makeEndpointInfo = standardEndpointInfo
	}
	return &EndpointSliceCache{
		sliceByServiceMap: map[types.NamespacedName]map[string]*endpointSliceInfo{},
		hostname:          hostname,
		isIPv6Mode:        isIPv6Mode,
		makeEndpointInfo:  makeEndpointInfo,
		recorder:          recorder,
	}
}

// standardEndpointInfo is the default makeEndpointFunc.
func standardEndpointInfo(ep *BaseEndpointInfo) Endpoint {
	return ep
}

// Update a slice in the cache.
func (cache *EndpointSliceCache) Update(endpointSlice *discovery.EndpointSlice) {
	serviceKey, sliceKey, err := endpointSliceCacheKeys(endpointSlice)
	if err != nil {
		klog.Warningf("Error getting endpoint slice cache keys: %v", err)
		return
	}

	esInfo := &endpointSliceInfo{
		Ports:     endpointSlice.Ports,
		Endpoints: []*endpointInfo{},
	}
	for _, endpoint := range endpointSlice.Endpoints {
		if endpoint.Conditions.Ready == nil || *endpoint.Conditions.Ready == true {
			esInfo.Endpoints = append(esInfo.Endpoints, &endpointInfo{
				Addresses: endpoint.Addresses,
				Topology:  endpoint.Topology,
			})
		}
	}
	if _, exists := cache.sliceByServiceMap[serviceKey]; !exists {
		cache.sliceByServiceMap[serviceKey] = map[string]*endpointSliceInfo{}
	}
	cache.sliceByServiceMap[serviceKey][sliceKey] = esInfo
}

// Delete a slice from the cache.
func (cache *EndpointSliceCache) Delete(endpointSlice *discovery.EndpointSlice) {
	serviceKey, sliceKey, err := endpointSliceCacheKeys(endpointSlice)
	if err != nil {
		klog.Warningf("Error getting endpoint slice cache keys: %v", err)
		return
	}
	delete(cache.sliceByServiceMap[serviceKey], sliceKey)
}

// EndpointsMap computes an EndpointsMap for a given service.
func (cache *EndpointSliceCache) EndpointsMap(serviceNN types.NamespacedName) EndpointsMap {
	endpointInfoBySP := cache.endpointInfoByServicePort(serviceNN)
	return endpointsMapFromEndpointInfo(endpointInfoBySP)
}

// endpointInfoByServicePort groups endpoint info by service port name and address.
func (cache *EndpointSliceCache) endpointInfoByServicePort(serviceNN types.NamespacedName) spToEndpointMap {
	endpointInfoBySP := spToEndpointMap{}
	sliceInfoByName, ok := cache.sliceByServiceMap[serviceNN]

	if !ok {
		return endpointInfoBySP
	}

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
		if cache.isIPv6Mode != nil && utilnet.IsIPv6String(endpoint.Addresses[0]) != *cache.isIPv6Mode {
			// Emit event on the corresponding service which had a different IP
			// version than the endpoint.
			utilproxy.LogAndEmitIncorrectIPVersionEvent(cache.recorder, "endpointslice", endpoint.Addresses[0], serviceNN.Namespace, serviceNN.Name, "")
			continue
		}

		isLocal := cache.isLocal(endpoint.Topology[v1.LabelHostname])
		endpointInfo := newBaseEndpointInfo(endpoint.Addresses[0], portNum, isLocal)

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
