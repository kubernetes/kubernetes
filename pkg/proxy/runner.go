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
	"time"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	proxyapis "k8s.io/kubernetes/pkg/proxy/apis"
	proxyconfig "k8s.io/kubernetes/pkg/proxy/config"
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

// StartInformers starts the runner's informers
func (r *Runner) StartInformers(client kubernetes.Interface, informerSyncPeriod time.Duration, nodeName string, useNodeCIDRInformer bool, podCIDRs []string) error {
	// NewRequirement can't return an error if you're passing it known-valid data
	noProxyName, _ := labels.NewRequirement(proxyapis.LabelServiceProxyName, selection.DoesNotExist, nil)
	noHeadlessEndpoints, _ := labels.NewRequirement(v1.IsHeadlessService, selection.DoesNotExist, nil)
	labelSelector := labels.NewSelector()
	labelSelector = labelSelector.Add(*noProxyName, *noHeadlessEndpoints)

	// Make informers that filter out objects that want a non-default service proxy.
	informerFactory := informers.NewSharedInformerFactoryWithOptions(client, informerSyncPeriod,
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			options.LabelSelector = labelSelector.String()
		}))

	// Create configs (i.e. Watches for Services and EndpointSlices)
	// Note: RegisterHandler() calls need to happen before creation of Sources because sources
	// only notify on changes, and the initial update (on process start) may be lost if no handlers
	// are registered yet.
	serviceConfig := proxyconfig.NewServiceConfig(informerFactory.Core().V1().Services(), informerSyncPeriod)
	serviceConfig.RegisterEventHandler(r)
	go serviceConfig.Run(wait.NeverStop)

	endpointSliceConfig := proxyconfig.NewEndpointSliceConfig(informerFactory.Discovery().V1().EndpointSlices(), informerSyncPeriod)
	endpointSliceConfig.RegisterEventHandler(r)
	go endpointSliceConfig.Run(wait.NeverStop)

	// This has to start after the calls to NewServiceConfig because that
	// function must configure its shared informer event handlers first.
	informerFactory.Start(wait.NeverStop)

	// Make an informer that selects for our nodename.
	currentNodeInformerFactory := informers.NewSharedInformerFactoryWithOptions(client, informerSyncPeriod,
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			options.FieldSelector = fields.OneTermEqualSelector("metadata.name", nodeName).String()
		}))
	nodeConfig := proxyconfig.NewNodeConfig(currentNodeInformerFactory.Core().V1().Nodes(), informerSyncPeriod)

	if useNodeCIDRInformer {
		nodeConfig.RegisterEventHandler(NewNodePodCIDRHandler(podCIDRs))
	}
	nodeConfig.RegisterEventHandler(r)

	go nodeConfig.Run(wait.NeverStop)

	// This has to start after the calls to NewNodeConfig because that must
	// configure the shared informer event handler first.
	currentNodeInformerFactory.Start(wait.NeverStop)

	return nil
}

// Run starts the main loop of the Runner (in other goroutines)
func (r *Runner) Run() {
	if r.ipv4Proxier != nil {
		go r.ipv4Proxier.SyncLoop()
	} else if r.ipv6Proxier != nil {
		go r.ipv6Proxier.SyncLoop()
	}
}

// ipv4Sync immediately synchronizes the IPv4 provider
func (r *Runner) ipv4Sync() {
	r.ipv4Proxier.Sync()
}

// ipv6Sync immediately synchronizes the IPv6 provider
func (r *Runner) ipv6Sync() {
	r.ipv6Proxier.Sync()
}

// OnServiceAdd is called whenever creation of new service object is observed.
func (r *Runner) OnServiceAdd(service *v1.Service) {
	if r.ipv4Proxier != nil {
		if r.ipv4Proxier.OnServiceAdd(service) {
			r.ipv4Sync()
		}
	}
	if r.ipv6Proxier != nil {
		if r.ipv6Proxier.OnServiceAdd(service) {
			r.ipv6Sync()
		}
	}
}

// OnServiceUpdate is called whenever modification of an existing
// service object is observed.
func (r *Runner) OnServiceUpdate(oldService, service *v1.Service) {
	if r.ipv4Proxier != nil {
		if r.ipv4Proxier.OnServiceUpdate(oldService, service) {
			r.ipv4Sync()
		}
	}
	if r.ipv6Proxier != nil {
		if r.ipv6Proxier.OnServiceUpdate(oldService, service) {
			r.ipv6Sync()
		}
	}
}

// OnServiceDelete is called whenever deletion of an existing service
// object is observed.
func (r *Runner) OnServiceDelete(service *v1.Service) {
	if r.ipv4Proxier != nil {
		if r.ipv4Proxier.OnServiceDelete(service) {
			r.ipv4Sync()
		}
	}
	if r.ipv6Proxier != nil {
		if r.ipv6Proxier.OnServiceDelete(service) {
			r.ipv6Sync()
		}
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
			if r.ipv4Proxier.OnEndpointSliceAdd(endpointSlice) {
				r.ipv4Sync()
			}
		}
	case discovery.AddressTypeIPv6:
		if r.ipv6Proxier != nil {
			if r.ipv6Proxier.OnEndpointSliceAdd(endpointSlice) {
				r.ipv6Sync()
			}
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
			if r.ipv4Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice) {
				r.ipv4Sync()
			}
		}
	case discovery.AddressTypeIPv6:
		if r.ipv6Proxier != nil {
			if r.ipv6Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice) {
				r.ipv6Sync()
			}
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
			if r.ipv4Proxier.OnEndpointSliceDelete(endpointSlice) {
				r.ipv4Sync()
			}
		}
	case discovery.AddressTypeIPv6:
		if r.ipv6Proxier != nil {
			if r.ipv6Proxier.OnEndpointSliceDelete(endpointSlice) {
				r.ipv6Sync()
			}
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
		if r.ipv4Proxier.OnNodeAdd(node) {
			r.ipv4Sync()
		}
	}
	if r.ipv6Proxier != nil {
		if r.ipv6Proxier.OnNodeAdd(node) {
			r.ipv6Sync()
		}
	}
}

// OnNodeUpdate is called whenever modification of an existing
// node object is observed.
func (r *Runner) OnNodeUpdate(oldNode, node *v1.Node) {
	if r.ipv4Proxier != nil {
		if r.ipv4Proxier.OnNodeUpdate(oldNode, node) {
			r.ipv4Sync()
		}
	}
	if r.ipv6Proxier != nil {
		if r.ipv6Proxier.OnNodeUpdate(oldNode, node) {
			r.ipv6Sync()
		}
	}
}

// OnNodeDelete is called whenever deletion of an existing node
// object is observed.
func (r *Runner) OnNodeDelete(node *v1.Node) {
	if r.ipv4Proxier != nil {
		if r.ipv4Proxier.OnNodeDelete(node) {
			r.ipv4Sync()
		}
	}
	if r.ipv6Proxier != nil {
		if r.ipv6Proxier.OnNodeDelete(node) {
			r.ipv6Sync()
		}
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
