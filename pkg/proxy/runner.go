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
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/klog/v2"
)

type Runner struct {
	ipv4Proxier Provider
	ipv6Proxier Provider
}

// NewRunner returns a proxy runner that dispatches to the underlying IPv4
// and/or IPv6 proxies.
func NewRunner(ipv4Proxier, ipv6Proxier Provider) *Runner {
	return &Runner{
		ipv4Proxier: ipv4Proxier,
		ipv6Proxier: ipv6Proxier,
	}
}

// Sync immediately synchronizes the providers' current states to the proxy rules.
func (r *Runner) Sync() {
	if r.ipv4Proxier != nil {
		r.ipv4Proxier.Sync()
	}
	if r.ipv6Proxier != nil {
		r.ipv6Proxier.Sync()
	}
}

// Run starts the main loop of the Runner (in other goroutines)
func (r *Runner) Run() {
	if r.ipv4Proxier != nil {
		go r.ipv4Proxier.SyncLoop()
	} else if r.ipv6Proxier != nil {
		go r.ipv6Proxier.SyncLoop()
	}
}

// OnServiceAdd is called whenever creation of new service object is observed.
func (r *Runner) OnServiceAdd(service *v1.Service) {
	if r.ipv4Proxier != nil {
		r.ipv4Proxier.OnServiceAdd(service)
	}
	if r.ipv6Proxier != nil {
		r.ipv6Proxier.OnServiceAdd(service)
	}
}

// OnServiceUpdate is called whenever modification of an existing
// service object is observed.
func (r *Runner) OnServiceUpdate(oldService, service *v1.Service) {
	if r.ipv4Proxier != nil {
		r.ipv4Proxier.OnServiceUpdate(oldService, service)
	}
	if r.ipv6Proxier != nil {
		r.ipv6Proxier.OnServiceUpdate(oldService, service)
	}
}

// OnServiceDelete is called whenever deletion of an existing service
// object is observed.
func (r *Runner) OnServiceDelete(service *v1.Service) {
	if r.ipv4Proxier != nil {
		r.ipv4Proxier.OnServiceDelete(service)
	}
	if r.ipv6Proxier != nil {
		r.ipv6Proxier.OnServiceDelete(service)
	}
}

// OnServiceSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (r *Runner) OnServiceSynced() {
	if r.ipv4Proxier != nil {
		r.ipv4Proxier.OnServiceSynced()
	}
	if r.ipv6Proxier != nil {
		r.ipv6Proxier.OnServiceSynced()
	}
}

// OnEndpointSliceAdd is called whenever creation of a new endpoint slice object
// is observed.
func (r *Runner) OnEndpointSliceAdd(endpointSlice *discovery.EndpointSlice) {
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		if r.ipv4Proxier != nil {
			r.ipv4Proxier.OnEndpointSliceAdd(endpointSlice)
		}
	case discovery.AddressTypeIPv6:
		if r.ipv6Proxier != nil {
			r.ipv6Proxier.OnEndpointSliceAdd(endpointSlice)
		}
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", endpointSlice.AddressType)
	}
}

// OnEndpointSliceUpdate is called whenever modification of an existing endpoint
// slice object is observed.
func (r *Runner) OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice *discovery.EndpointSlice) {
	switch newEndpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		if r.ipv4Proxier != nil {
			r.ipv4Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
		}
	case discovery.AddressTypeIPv6:
		if r.ipv6Proxier != nil {
			r.ipv6Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
		}
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", newEndpointSlice.AddressType)
	}
}

// OnEndpointSliceDelete is called whenever deletion of an existing endpoint slice
// object is observed.
func (r *Runner) OnEndpointSliceDelete(endpointSlice *discovery.EndpointSlice) {
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		if r.ipv4Proxier != nil {
			r.ipv4Proxier.OnEndpointSliceDelete(endpointSlice)
		}
	case discovery.AddressTypeIPv6:
		if r.ipv6Proxier != nil {
			r.ipv6Proxier.OnEndpointSliceDelete(endpointSlice)
		}
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", endpointSlice.AddressType)
	}
}

// OnEndpointSlicesSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (r *Runner) OnEndpointSlicesSynced() {
	if r.ipv4Proxier != nil {
		r.ipv4Proxier.OnEndpointSlicesSynced()
	}
	if r.ipv6Proxier != nil {
		r.ipv6Proxier.OnEndpointSlicesSynced()
	}
}

// OnNodeAdd is called whenever creation of new node object is observed.
func (r *Runner) OnNodeAdd(node *v1.Node) {
	if r.ipv4Proxier != nil {
		r.ipv4Proxier.OnNodeAdd(node)
	}
	if r.ipv6Proxier != nil {
		r.ipv6Proxier.OnNodeAdd(node)
	}
}

// OnNodeUpdate is called whenever modification of an existing
// node object is observed.
func (r *Runner) OnNodeUpdate(oldNode, node *v1.Node) {
	if r.ipv4Proxier != nil {
		r.ipv4Proxier.OnNodeUpdate(oldNode, node)
	}
	if r.ipv6Proxier != nil {
		r.ipv6Proxier.OnNodeUpdate(oldNode, node)
	}
}

// OnNodeDelete is called whenever deletion of an existing node
// object is observed.
func (r *Runner) OnNodeDelete(node *v1.Node) {
	if r.ipv4Proxier != nil {
		r.ipv4Proxier.OnNodeDelete(node)
	}
	if r.ipv6Proxier != nil {
		r.ipv6Proxier.OnNodeDelete(node)
	}
}

// OnNodeSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (r *Runner) OnNodeSynced() {
	if r.ipv4Proxier != nil {
		r.ipv4Proxier.OnNodeSynced()
	}
	if r.ipv6Proxier != nil {
		r.ipv6Proxier.OnNodeSynced()
	}
}
