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
	"reflect"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	proxyapis "k8s.io/kubernetes/pkg/proxy/apis"
	proxyconfig "k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	"k8s.io/kubernetes/pkg/util/async"
)

type Runner struct {
	sync.Mutex

	ipv4Proxier         Provider
	ipv4Runner          *async.BoundedFrequencyRunner
	ipv4ServiceTracker  *ServiceChangeTracker
	ipv4EndpointTracker *EndpointChangeTracker

	ipv6Proxier         Provider
	ipv6Runner          *async.BoundedFrequencyRunner
	ipv6ServiceTracker  *ServiceChangeTracker
	ipv6EndpointTracker *EndpointChangeTracker

	syncPeriod    time.Duration
	minSyncPeriod time.Duration
	healthzServer healthcheck.ProxierHealthUpdater

	informerWaiters []cache.InformerSynced
	nodeLabels      map[string]string
}

// NewRunner returns a proxy runner that dispatches to the underlying IPv4
// and/or IPv6 proxies.
func NewRunner(
	ipv4Proxier   Provider,
	ipv6Proxier   Provider,
	syncPeriod    time.Duration,
	minSyncPeriod time.Duration,
	healthzServer healthcheck.ProxierHealthUpdater,
) *Runner {
	r := &Runner{
		ipv4Proxier:   ipv4Proxier,
		ipv6Proxier:   ipv6Proxier,
		syncPeriod:    syncPeriod,
		minSyncPeriod: syncPeriod,
		healthzServer: healthzServer,
	}

	if ipv4Proxier != nil {
		r.ipv4ServiceTracker = ipv4Proxier.MakeServiceChangeTracker()
		r.ipv4EndpointTracker = ipv4Proxier.MakeEndpointChangeTracker()

		r.ipv4Runner = async.NewBoundedFrequencyRunner("ipv4Runner", r.ipv4SyncNow, minSyncPeriod, time.Hour, 2)
		// Queue a sync to occur as soon as the runner's loop is started
		r.ipv4Runner.Run()
	}
	if ipv6Proxier != nil {
		r.ipv6ServiceTracker = ipv6Proxier.MakeServiceChangeTracker()
		r.ipv6EndpointTracker = ipv6Proxier.MakeEndpointChangeTracker()

		r.ipv6Runner = async.NewBoundedFrequencyRunner("ipv6Runner", r.ipv6SyncNow, minSyncPeriod, time.Hour, 2)
		// Queue a sync to occur as soon as the runner's loop is started
		r.ipv6Runner.Run()
	}

	return r
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
	serviceInformer := informerFactory.Core().V1().Services()
	serviceConfig := proxyconfig.NewServiceConfig(serviceInformer, informerSyncPeriod)
	serviceConfig.RegisterEventHandler(r)
	r.informerWaiters = append(r.informerWaiters, serviceInformer.Informer().HasSynced)

	endpointSliceInformer := informerFactory.Discovery().V1().EndpointSlices()
	endpointSliceConfig := proxyconfig.NewEndpointSliceConfig(endpointSliceInformer, informerSyncPeriod)
	endpointSliceConfig.RegisterEventHandler(r)
	r.informerWaiters = append(r.informerWaiters, endpointSliceInformer.Informer().HasSynced)

	// This has to start after the calls to NewServiceConfig because that
	// function must configure its shared informer event handlers first.
	informerFactory.Start(wait.NeverStop)

	// Make an informer that selects for our nodename.
	currentNodeInformerFactory := informers.NewSharedInformerFactoryWithOptions(client, informerSyncPeriod,
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			options.FieldSelector = fields.OneTermEqualSelector("metadata.name", nodeName).String()
		}))
	nodeInformer := currentNodeInformerFactory.Core().V1().Nodes()
	nodeConfig := proxyconfig.NewNodeConfig(nodeInformer, informerSyncPeriod)
	r.informerWaiters = append(r.informerWaiters, nodeInformer.Informer().HasSynced)

	if useNodeCIDRInformer {
		nodeConfig.RegisterEventHandler(NewNodePodCIDRHandler(podCIDRs))
	}
	nodeConfig.RegisterEventHandler(r)

	// This has to start after the calls to NewNodeConfig because that must
	// configure the shared informer event handler first.
	currentNodeInformerFactory.Start(wait.NeverStop)

	return nil
}

// Run starts the main loop of the Runner (in other goroutines)
func (r *Runner) Run() {
	go r.waitAndRun()
}

func (r *Runner) waitAndRun() {
	klog.InfoS("Waiting for proxy informers to sync")
	cache.WaitForNamedCacheSync("proxy.Runner", wait.NeverStop, r.informerWaiters...)
	klog.InfoS("Proxy informers are synced")

	r.healthzServer.Updated()
	metrics.SyncProxyRulesLastQueuedTimestamp.SetToCurrentTime()

	if r.ipv4Runner != nil {
		go r.ipv4Runner.Loop(wait.NeverStop)
	}
	if r.ipv6Runner != nil {
		go r.ipv6Runner.Loop(wait.NeverStop)
	}
}

// ipv4SyncNow immediately synchronizes the IPv4 provider
func (r *Runner) ipv4SyncNow() {
	r.Lock()
	defer r.Unlock()

	// Keep track of how long syncs take.
	start := time.Now()
	defer func() {
		metrics.SyncProxyRulesLatency.Observe(metrics.SinceInSeconds(start))
		klog.V(2).InfoS("Syncing proxy rules complete", "elapsed", time.Since(start))
	}()

	switch r.ipv4Proxier.Sync(r.ipv4ServiceTracker, r.ipv4EndpointTracker, r.nodeLabels) {
	case SyncSuccess:
		r.healthzServer.Updated()
		metrics.SyncProxyRulesLastTimestamp.SetToCurrentTime()

	case SyncFailure:

	case SyncRetry:
		klog.InfoS("Sync failed", "retryingTime", r.syncPeriod)
		r.ipv4Runner.RetryAfter(r.syncPeriod)
	}
}

// ipv4Sync queues a sync of the IPv4 provider
func (r *Runner) ipv4Sync() {
	r.healthzServer.QueuedUpdate()
	metrics.SyncProxyRulesLastQueuedTimestamp.SetToCurrentTime()

	r.ipv4Runner.Run()
}

// ipv6SyncNow immediately synchronizes the IPv6 provider
func (r *Runner) ipv6SyncNow() {
	r.Lock()
	defer r.Unlock()

	// Keep track of how long syncs take.
	start := time.Now()
	defer func() {
		metrics.SyncProxyRulesLatency.Observe(metrics.SinceInSeconds(start))
		klog.V(4).InfoS("Syncing proxy rules complete", "elapsed", time.Since(start))
	}()

	switch r.ipv6Proxier.Sync(r.ipv4ServiceTracker, r.ipv4EndpointTracker, r.nodeLabels) {
	case SyncSuccess:
		r.healthzServer.Updated()
		metrics.SyncProxyRulesLastTimestamp.SetToCurrentTime()

	case SyncFailure:

	case SyncRetry:
		klog.InfoS("Sync failed", "retryingTime", r.syncPeriod)
		r.ipv6Runner.RetryAfter(r.syncPeriod)
	}
}

// ipv6Sync queues a sync of the IPv4 provider
func (r *Runner) ipv6Sync() {
	r.healthzServer.QueuedUpdate()
	metrics.SyncProxyRulesLastQueuedTimestamp.SetToCurrentTime()

	r.ipv6Runner.Run()
}

// OnServiceAdd is called whenever creation of new service object is observed.
func (r *Runner) OnServiceAdd(service *v1.Service) {
	r.OnServiceUpdate(nil, service)
}

// OnServiceUpdate is called whenever modification of an existing
// service object is observed.
func (r *Runner) OnServiceUpdate(oldService, service *v1.Service) {
	if r.ipv4Proxier != nil {
		if r.ipv4ServiceTracker.Update(oldService, service) {
			r.ipv4Sync()
		}
	}
	if r.ipv6Proxier != nil {
		if r.ipv6ServiceTracker.Update(oldService, service) {
			r.ipv6Sync()
		}
	}
}

// OnServiceDelete is called whenever deletion of an existing service
// object is observed.
func (r *Runner) OnServiceDelete(service *v1.Service) {
	r.OnServiceUpdate(service, nil)
}

// OnEndpointSliceAdd is called whenever creation of a new endpoint slice object
// is observed.
func (r *Runner) OnEndpointSliceAdd(endpointSlice *discovery.EndpointSlice) {
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		if r.ipv4Proxier != nil {
			if r.ipv4EndpointTracker.EndpointSliceUpdate(endpointSlice, false) {
				r.ipv4Sync()
			}
		}
	case discovery.AddressTypeIPv6:
		if r.ipv6Proxier != nil {
			if r.ipv6EndpointTracker.EndpointSliceUpdate(endpointSlice, false) {
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
			if r.ipv4EndpointTracker.EndpointSliceUpdate(newEndpointSlice, false) {
				r.ipv4Sync()
			}
		}
	case discovery.AddressTypeIPv6:
		if r.ipv6Proxier != nil {
			if r.ipv6EndpointTracker.EndpointSliceUpdate(newEndpointSlice, false) {
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
			if r.ipv4EndpointTracker.EndpointSliceUpdate(endpointSlice, true) {
				r.ipv4Sync()
			}
		}
	case discovery.AddressTypeIPv6:
		if r.ipv6Proxier != nil {
			if r.ipv6EndpointTracker.EndpointSliceUpdate(endpointSlice, true) {
				r.ipv6Sync()
			}
		}
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", endpointSlice.AddressType)
	}
}

// OnNodeAdd is called whenever creation of new node object is observed.
func (r *Runner) OnNodeAdd(node *v1.Node) {
	r.OnNodeUpdate(nil, node)
}

// OnNodeUpdate is called whenever modification of an existing
// node object is observed.
func (r *Runner) OnNodeUpdate(_, node *v1.Node) {
	if reflect.DeepEqual(r.nodeLabels, node.Labels) {
		return
	}

	r.Lock()
	defer r.Unlock()
	r.nodeLabels = map[string]string{}
	for k, v := range node.Labels {
		r.nodeLabels[k] = v
	}
	klog.V(4).InfoS("Updated proxier node labels", "labels", node.Labels)

	// FIXME proxier.needFullSync = true
	if r.ipv4Proxier != nil {
		r.ipv4Sync()
	}
	if r.ipv6Proxier != nil {
		r.ipv6Sync()
	}
}

// OnNodeDelete is called whenever deletion of an existing node
// object is observed.
func (r *Runner) OnNodeDelete(node *v1.Node) {
	r.Lock()
	defer r.Unlock()
	r.nodeLabels = map[string]string{}

	// FIXME proxier.needFullSync = true
	if r.ipv4Proxier != nil {
		r.ipv4Sync()
	}
	if r.ipv6Proxier != nil {
		r.ipv6Sync()
	}
}
