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
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/config"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"

	utilnet "k8s.io/utils/net"

	discovery "k8s.io/api/discovery/v1beta1"
)

type metaProxier struct {
	ipv4Proxier proxy.Provider
	ipv6Proxier proxy.Provider
	// TODO(imroc): implement node handler for meta proxier.
	config.NoopNodeHandler
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
	if utilproxy.ShouldSkipService(service) {
		return
	}
	if utilnet.IsIPv6String(service.Spec.ClusterIP) {
		proxier.ipv6Proxier.OnServiceAdd(service)
	} else {
		proxier.ipv4Proxier.OnServiceAdd(service)
	}
}

// OnServiceUpdate is called whenever modification of an existing
// service object is observed.
func (proxier *metaProxier) OnServiceUpdate(oldService, service *v1.Service) {
	if utilproxy.ShouldSkipService(service) {
		return
	}
	// IPFamily is immutable, hence we only need to check on the new service
	if utilnet.IsIPv6String(service.Spec.ClusterIP) {
		proxier.ipv6Proxier.OnServiceUpdate(oldService, service)
	} else {
		proxier.ipv4Proxier.OnServiceUpdate(oldService, service)
	}
}

// OnServiceDelete is called whenever deletion of an existing service
// object is observed.
func (proxier *metaProxier) OnServiceDelete(service *v1.Service) {
	if utilproxy.ShouldSkipService(service) {
		return
	}
	if utilnet.IsIPv6String(service.Spec.ClusterIP) {
		proxier.ipv6Proxier.OnServiceDelete(service)
	} else {
		proxier.ipv4Proxier.OnServiceDelete(service)
	}
}

// OnServiceSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *metaProxier) OnServiceSynced() {
	proxier.ipv4Proxier.OnServiceSynced()
	proxier.ipv6Proxier.OnServiceSynced()
}

// OnEndpointsAdd is called whenever creation of new endpoints object
// is observed.
func (proxier *metaProxier) OnEndpointsAdd(endpoints *v1.Endpoints) {
	ipFamily, err := endpointsIPFamily(endpoints)
	if err != nil {
		klog.V(4).Infof("failed to add endpoints %s/%s with error %v", endpoints.ObjectMeta.Namespace, endpoints.ObjectMeta.Name, err)
		return
	}
	if *ipFamily == v1.IPv4Protocol {
		proxier.ipv4Proxier.OnEndpointsAdd(endpoints)
		return
	}
	proxier.ipv6Proxier.OnEndpointsAdd(endpoints)
}

// OnEndpointsUpdate is called whenever modification of an existing
// endpoints object is observed.
func (proxier *metaProxier) OnEndpointsUpdate(oldEndpoints, endpoints *v1.Endpoints) {
	ipFamily, err := endpointsIPFamily(endpoints)
	if err != nil {
		klog.V(4).Infof("failed to update endpoints %s/%s with error %v", endpoints.ObjectMeta.Namespace, endpoints.ObjectMeta.Name, err)
		return
	}

	if *ipFamily == v1.IPv4Protocol {
		proxier.ipv4Proxier.OnEndpointsUpdate(oldEndpoints, endpoints)
		return
	}
	proxier.ipv6Proxier.OnEndpointsUpdate(oldEndpoints, endpoints)
}

// OnEndpointsDelete is called whenever deletion of an existing
// endpoints object is observed.
func (proxier *metaProxier) OnEndpointsDelete(endpoints *v1.Endpoints) {
	ipFamily, err := endpointsIPFamily(endpoints)
	if err != nil {
		klog.V(4).Infof("failed to delete endpoints %s/%s with error %v", endpoints.ObjectMeta.Namespace, endpoints.ObjectMeta.Name, err)
		return
	}

	if *ipFamily == v1.IPv4Protocol {
		proxier.ipv4Proxier.OnEndpointsDelete(endpoints)
		return
	}
	proxier.ipv6Proxier.OnEndpointsDelete(endpoints)
}

// OnEndpointsSynced is called once all the initial event handlers
// were called and the state is fully propagated to local cache.
func (proxier *metaProxier) OnEndpointsSynced() {
	proxier.ipv4Proxier.OnEndpointsSynced()
	proxier.ipv6Proxier.OnEndpointsSynced()
}

// TODO: (khenidak) implement EndpointSlice handling

// OnEndpointSliceAdd is called whenever creation of a new endpoint slice object
// is observed.
func (proxier *metaProxier) OnEndpointSliceAdd(endpointSlice *discovery.EndpointSlice) {
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		proxier.ipv4Proxier.OnEndpointSliceAdd(endpointSlice)
	case discovery.AddressTypeIPv6:
		proxier.ipv6Proxier.OnEndpointSliceAdd(endpointSlice)
	default:
		klog.V(4).Infof("EndpointSlice address type not supported by kube-proxy: %s", endpointSlice.AddressType)
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
		klog.V(4).Infof("EndpointSlice address type not supported by kube-proxy: %s", newEndpointSlice.AddressType)
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
		klog.V(4).Infof("EndpointSlice address type not supported by kube-proxy: %s", endpointSlice.AddressType)
	}
}

// OnEndpointSlicesSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *metaProxier) OnEndpointSlicesSynced() {
	proxier.ipv4Proxier.OnEndpointSlicesSynced()
	proxier.ipv6Proxier.OnEndpointSlicesSynced()
}

// endpointsIPFamily that returns IPFamily of endpoints or error if
// failed to identify the IP family.
func endpointsIPFamily(endpoints *v1.Endpoints) (*v1.IPFamily, error) {
	if len(endpoints.Subsets) == 0 {
		return nil, fmt.Errorf("failed to identify ipfamily for endpoints (no subsets)")
	}

	// we only need to work with subset [0],endpoint controller
	// ensures that endpoints selected are of the same family.
	subset := endpoints.Subsets[0]
	if len(subset.Addresses) == 0 {
		return nil, fmt.Errorf("failed to identify ipfamily for endpoints (no addresses)")
	}
	// same apply on addresses
	address := subset.Addresses[0]
	if len(address.IP) == 0 {
		return nil, fmt.Errorf("failed to identify ipfamily for endpoints (address has no ip)")
	}

	ipv4 := v1.IPv4Protocol
	ipv6 := v1.IPv6Protocol
	if utilnet.IsIPv6String(address.IP) {
		return &ipv6, nil
	}

	return &ipv4, nil
}
