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
	discovery "k8s.io/api/discovery/v1beta1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/config"

	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	utilnet "k8s.io/utils/net"
)

type metaProxier struct {
	// actual, wrapped
	ipv4Proxier proxy.Provider
	// actual, wrapped
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

// getProxierByIPFamily returns the proxy selected for a specific ipfamily
func (proxier *metaProxier) getProxierByIPFamily(ipFamily v1.IPFamily) proxy.Provider {
	if ipFamily == v1.IPv4Protocol {
		return proxier.ipv4Proxier
	}

	return proxier.ipv6Proxier
}

//getProxierByClusterIP gets proxy by using identifying the ipFamily of ClusterIP
func (proxier *metaProxier) getProxierByClusterIP(service *v1.Service) proxy.Provider {
	ipFamily := v1.IPv4Protocol
	if utilnet.IsIPv6String(service.Spec.ClusterIP) {
		ipFamily = v1.IPv6Protocol
	}

	return proxier.getProxierByIPFamily(ipFamily)
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

	// this allows skew between new proxy and old apiserver
	if len(service.Spec.IPFamilies) == 0 {
		actual := proxier.getProxierByClusterIP(service)
		actual.OnServiceAdd(service)
		return
	}

	for _, ipFamily := range service.Spec.IPFamilies {
		actual := proxier.getProxierByIPFamily(ipFamily)
		actual.OnServiceAdd(service)
	}
}

// OnServiceUpdate is called whenever modification of an existing
// service object is observed.
func (proxier *metaProxier) OnServiceUpdate(oldService, service *v1.Service) {
	if utilproxy.ShouldSkipService(service) {
		return
	}

	// case zero: this allows skew between new proxy and old apiserver
	if len(service.Spec.IPFamilies) == 0 {
		actual := proxier.getProxierByClusterIP(oldService)
		actual.OnServiceUpdate(oldService, service)
		return
	}

	// case one: something has changed, but not families
	// call update on all families the service carries.
	if len(oldService.Spec.IPFamilies) == len(service.Spec.IPFamilies) {
		for _, ipFamily := range service.Spec.IPFamilies {
			actual := proxier.getProxierByIPFamily(ipFamily)
			actual.OnServiceUpdate(oldService, service)
		}

		klog.V(4).Infof("service %s/%s has been updated but no ip family change detected", service.Namespace, service.Name)
		return
	}
	// while apiserver does not allow changing primary ipfamily
	// we use the below approach to stay on the safe side.
	// note: in all cases, we check all families just
	// in case the service moved from ExternalName => ClusterIP

	// case two: service was upgraded (+1 ipFamily)
	// call add for new family
	// call update for existing family
	// note: Service might have been upgraded and
	// had port/toplogy keys etc  changed.
	if len(service.Spec.IPFamilies) > len(oldService.Spec.IPFamilies) {
		found := false
		for _, newSvcIPFamily := range service.Spec.IPFamilies {
			for _, existingSvcIPFamily := range oldService.Spec.IPFamilies {
				if newSvcIPFamily == existingSvcIPFamily {
					found = true
					break
				}
			}

			actual := proxier.getProxierByIPFamily(newSvcIPFamily)
			if found {
				actual.OnServiceUpdate(oldService, service)
			} else {
				klog.V(4).Infof("service %s/%s has been updated and ipfamily %v was added", service.Namespace, service.Name, newSvcIPFamily)
				actual.OnServiceAdd(service)
			}
		}

		return
	}

	// case three: service was downgraded
	// call delete for removed family
	// call update for existing family
	if len(service.Spec.IPFamilies) < len(oldService.Spec.IPFamilies) {
		found := false
		for _, existingSvcIPFamily := range oldService.Spec.IPFamilies {
			for _, newSvcIPFamily := range service.Spec.IPFamilies {
				if newSvcIPFamily == existingSvcIPFamily {
					found = true
					break
				}
			}

			actual := proxier.getProxierByIPFamily(existingSvcIPFamily)
			if found {
				actual.OnServiceUpdate(oldService, service)
			} else {
				klog.V(4).Infof("service %s/%s has been updated and ipfamily %v was was removed", service.Namespace, service.Name, existingSvcIPFamily)
				actual.OnServiceDelete(service)
			}
		}

		return
	}
}

// OnServiceDelete is called whenever deletion of an existing service
// object is observed.
func (proxier *metaProxier) OnServiceDelete(service *v1.Service) {
	if utilproxy.ShouldSkipService(service) {
		return
	}

	// this allows skew between new proxy and old apiserver
	if len(service.Spec.IPFamilies) == 0 {
		actual := proxier.getProxierByClusterIP(service)
		actual.OnServiceDelete(service)
		return
	}

	for _, ipFamily := range service.Spec.IPFamilies {
		actual := proxier.getProxierByIPFamily(ipFamily)
		actual.OnServiceDelete(service)
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
