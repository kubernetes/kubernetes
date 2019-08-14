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
	"k8s.io/api/core/v1"
	"k8s.io/klog"
	"net"
)

type MetaProxier struct {
	ipv4Proxier   ProxyProvider
	ipv6Proxier   ProxyProvider
	defaultFamily IpFamily
}

type IpFamily int

const (
	FamilyUnknown = iota + 1
	FamilyIpv4
	FamilyIpv6
)

// NewMetaProxier returns a dual-stack "meta-proxier". Proxier API
// calls will be dispatched to the ProxyProvider instances depending
// on address family.
func NewMetaProxier(ipv4Proxier, ipv6Proxier ProxyProvider, defaultFamily IpFamily) *MetaProxier {
	return &MetaProxier{
		ipv4Proxier:   ipv4Proxier,
		ipv6Proxier:   ipv6Proxier,
		defaultFamily: defaultFamily,
	}
}

// Sync immediately synchronizes the ProxyProvider's current state to
// proxy rules.
func (proxier *MetaProxier) Sync() {
	proxier.ipv4Proxier.Sync()
	proxier.ipv6Proxier.Sync()
}

// SyncLoop runs periodic work.  This is expected to run as a
// goroutine or as the main loop of the app.  It does not return.
func (proxier *MetaProxier) SyncLoop() {
	go proxier.ipv6Proxier.SyncLoop() // Use go-routine here!
	proxier.ipv4Proxier.SyncLoop()    // never returns
}

// OnServiceAdd is called whenever creation of new service object is observed.
func (proxier *MetaProxier) OnServiceAdd(service *v1.Service) {
	family := proxier.serviceFamily(service)
	switch family {
	case FamilyIpv4:
		proxier.ipv4Proxier.OnServiceAdd(service)
	case FamilyIpv6:
		proxier.ipv6Proxier.OnServiceAdd(service)
	}
}

// OnServiceUpdate is called whenever modification of an existing
// service object is observed.
func (proxier *MetaProxier) OnServiceUpdate(oldService, service *v1.Service) {
	family := proxier.serviceFamily(service)
	if family == FamilyUnknown {
		family = proxier.serviceFamily(oldService)
	}
	switch family {
	case FamilyIpv4:
		proxier.ipv4Proxier.OnServiceUpdate(oldService, service)
	case FamilyIpv6:
		proxier.ipv6Proxier.OnServiceUpdate(oldService, service)
	}
}

// OnServiceDelete is called whenever deletion of an existing service
// object is observed.
func (proxier *MetaProxier) OnServiceDelete(service *v1.Service) {
	switch proxier.serviceFamily(service) {
	case FamilyIpv4:
		proxier.ipv4Proxier.OnServiceDelete(service)
	case FamilyIpv6:
		proxier.ipv6Proxier.OnServiceDelete(service)
	}
}

// OnServiceSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *MetaProxier) OnServiceSynced() {
	proxier.ipv4Proxier.OnServiceSynced()
	proxier.ipv6Proxier.OnServiceSynced()
}

func (proxier *MetaProxier) serviceFamily(service *v1.Service) IpFamily {
	if service.Spec.ClusterIP == "" {
		return FamilyUnknown
	}
	if net.ParseIP(service.Spec.ClusterIP).To4() == nil {
		return FamilyIpv6
	}
	return FamilyIpv4
}

// OnEndpointsAdd is called whenever creation of new endpoints object
// is observed.
func (proxier *MetaProxier) OnEndpointsAdd(endpoints *v1.Endpoints) {
	switch endpointsFamily(endpoints) {
	case FamilyIpv4:
		proxier.ipv4Proxier.OnEndpointsAdd(endpoints)
	case FamilyIpv6:
		proxier.ipv6Proxier.OnEndpointsAdd(endpoints)
	}
}

// OnEndpointsUpdate is called whenever modification of an existing
// endpoints object is observed.
func (proxier *MetaProxier) OnEndpointsUpdate(oldEndpoints, endpoints *v1.Endpoints) {
	family := endpointsFamily(endpoints)
	if family == FamilyUnknown {
		family = endpointsFamily(oldEndpoints)
	}
	switch family {
	case FamilyIpv4:
		proxier.ipv4Proxier.OnEndpointsUpdate(oldEndpoints, endpoints)
	case FamilyIpv6:
		proxier.ipv6Proxier.OnEndpointsUpdate(oldEndpoints, endpoints)
	}
}

// OnEndpointsDelete is called whenever deletion of an existing
// endpoints object is observed.
func (proxier *MetaProxier) OnEndpointsDelete(endpoints *v1.Endpoints) {
	switch endpointsFamily(endpoints) {
	case FamilyIpv4:
		proxier.ipv4Proxier.OnEndpointsDelete(endpoints)
	case FamilyIpv6:
		proxier.ipv6Proxier.OnEndpointsDelete(endpoints)
	}
}

// OnEndpointsSynced is called once all the initial event handlers
// were called and the state is fully propagated to local cache.
func (proxier *MetaProxier) OnEndpointsSynced() {
	proxier.ipv4Proxier.OnEndpointsSynced()
	proxier.ipv6Proxier.OnEndpointsSynced()
}

func endpointsFamily(endpoints *v1.Endpoints) IpFamily {
	if endpoints.Subsets == nil || len(endpoints.Subsets) == 0 {
		return FamilyUnknown
	}
	for _, subset := range endpoints.Subsets {
		if family := endpointsSubsetFamily(subset); family != FamilyUnknown {
			return family
		}
	}
	klog.Warningf("Endpoints familyUnknown; %s:%s\n", endpoints.ObjectMeta.Namespace, endpoints.ObjectMeta.Name)
	return FamilyUnknown
}
func endpointsSubsetFamily(subsets v1.EndpointSubset) IpFamily {
	if family := endpointsAddressesFamily(subsets.Addresses); family != FamilyUnknown {
		return family
	}
	return endpointsAddressesFamily(subsets.NotReadyAddresses)
}
func endpointsAddressesFamily(addresses []v1.EndpointAddress) IpFamily {
	if addresses == nil || len(addresses) == 0 {
		return FamilyUnknown
	}
	for _, adr := range addresses {
		if adr.IP != "" {
			if net.ParseIP(adr.IP).To4() == nil {
				return FamilyIpv6
			} else {
				return FamilyIpv4
			}
		}
	}
	return FamilyUnknown
}
