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

package metaproxier

import (
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
)

type metaProxier struct {
	// actual, wrapped
	ipv4Proxier proxy.Provider
	// actual, wrapped
	ipv6Proxier proxy.Provider
}

// NewMetaProxier returns a dual-stack "meta-proxier". Proxier API
// calls will be dispatched to the ProxyProvider instances depending
// on address family.
func NewMetaProxier(ipv4Proxier, ipv6Proxier proxy.Provider) proxy.Provider {
	return proxy.Provider(&metaProxier{
		ipv4Proxier: ipv4Proxier,
		ipv6Proxier: ipv6Proxier,
	})
}

// Sync immediately synchronizes the ProxyProvider's current state to
// proxy rules.
func (proxier *metaProxier) Sync() {
	proxier.ipv4Proxier.Sync()
	proxier.ipv6Proxier.Sync()
}

// SyncLoop runs periodic work.  This is expected to run as a
// goroutine or as the main loop of the app.  It does not return.
func (proxier *metaProxier) SyncLoop() {
	go proxier.ipv6Proxier.SyncLoop() // Use go-routine here!
	proxier.ipv4Proxier.SyncLoop()    // never returns
}

// OnServiceAdd is called whenever creation of new service object is observed.
func (proxier *metaProxier) OnServiceAdd(service *v1.Service) {
	proxier.ipv4Proxier.OnServiceAdd(service)
	proxier.ipv6Proxier.OnServiceAdd(service)
}

// OnServiceUpdate is called whenever modification of an existing
// service object is observed.
func (proxier *metaProxier) OnServiceUpdate(oldService, service *v1.Service) {
	proxier.ipv4Proxier.OnServiceUpdate(oldService, service)
	proxier.ipv6Proxier.OnServiceUpdate(oldService, service)
}

// OnServiceDelete is called whenever deletion of an existing service
// object is observed.
func (proxier *metaProxier) OnServiceDelete(service *v1.Service) {
	proxier.ipv4Proxier.OnServiceDelete(service)
	proxier.ipv6Proxier.OnServiceDelete(service)

}

// OnServiceSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *metaProxier) OnServiceSynced() {
	proxier.ipv4Proxier.OnServiceSynced()
	proxier.ipv6Proxier.OnServiceSynced()
}

// OnEndpointSliceAdd is called whenever creation of a new endpoint slice object
// is observed.
func (proxier *metaProxier) OnEndpointSliceAdd(endpointSlice *discovery.EndpointSlice) {
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		proxier.ipv4Proxier.OnEndpointSliceAdd(endpointSlice)
	case discovery.AddressTypeIPv6:
		proxier.ipv6Proxier.OnEndpointSliceAdd(endpointSlice)
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", endpointSlice.AddressType)
	}
}

// OnEndpointSliceUpdate is called whenever modification of an existing endpoint
// slice object is observed.
func (proxier *metaProxier) OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice *discovery.EndpointSlice) {
	switch newEndpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		proxier.ipv4Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
	case discovery.AddressTypeIPv6:
		proxier.ipv6Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", newEndpointSlice.AddressType)
	}
}

// OnEndpointSliceDelete is called whenever deletion of an existing endpoint slice
// object is observed.
func (proxier *metaProxier) OnEndpointSliceDelete(endpointSlice *discovery.EndpointSlice) {
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		proxier.ipv4Proxier.OnEndpointSliceDelete(endpointSlice)
	case discovery.AddressTypeIPv6:
		proxier.ipv6Proxier.OnEndpointSliceDelete(endpointSlice)
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", endpointSlice.AddressType)
	}
}

// OnEndpointSlicesSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *metaProxier) OnEndpointSlicesSynced() {
	proxier.ipv4Proxier.OnEndpointSlicesSynced()
	proxier.ipv6Proxier.OnEndpointSlicesSynced()
}

// OnNodeAdd is called whenever creation of new node object is observed.
func (proxier *metaProxier) OnNodeAdd(node *v1.Node) {
	proxier.ipv4Proxier.OnNodeAdd(node)
	proxier.ipv6Proxier.OnNodeAdd(node)
}

// OnNodeUpdate is called whenever modification of an existing
// node object is observed.
func (proxier *metaProxier) OnNodeUpdate(oldNode, node *v1.Node) {
	proxier.ipv4Proxier.OnNodeUpdate(oldNode, node)
	proxier.ipv6Proxier.OnNodeUpdate(oldNode, node)
}

// OnNodeDelete is called whenever deletion of an existing node
// object is observed.
func (proxier *metaProxier) OnNodeDelete(node *v1.Node) {
	proxier.ipv4Proxier.OnNodeDelete(node)
	proxier.ipv6Proxier.OnNodeDelete(node)

}

// OnNodeSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *metaProxier) OnNodeSynced() {
	proxier.ipv4Proxier.OnNodeSynced()
	proxier.ipv6Proxier.OnNodeSynced()
}

// OnServiceCIDRsChanged is called whenever a change is observed
// in any of the ServiceCIDRs, and provides complete list of service cidrs.
func (proxier *metaProxier) OnServiceCIDRsChanged(cidrs []string) {
	proxier.ipv4Proxier.OnServiceCIDRsChanged(cidrs)
	proxier.ipv6Proxier.OnServiceCIDRsChanged(cidrs)
}
