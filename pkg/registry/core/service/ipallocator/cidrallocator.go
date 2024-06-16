/*
Copyright 2023 The Kubernetes Authors.

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

package ipallocator

import (
	"fmt"
	"net"
	"net/netip"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingv1alpha1 "k8s.io/api/networking/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	networkingv1alpha1informers "k8s.io/client-go/informers/networking/v1alpha1"
	networkingv1alpha1client "k8s.io/client-go/kubernetes/typed/networking/v1alpha1"
	networkingv1alpha1listers "k8s.io/client-go/listers/networking/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/api/servicecidr"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	netutils "k8s.io/utils/net"
)

// MetaAllocator maintains a structure with IP alloctors for the corresponding ServiceCIDRs.
// CIDR overlapping is allowed and the MetaAllocator should take this into consideration.
// Each allocator doesn't stored the IPAddresses, instead it reads them from the informer
// cache, it is cheap to create and delete IP Allocators.
// MetaAllocator use any READY allocator to Allocate IP addresses that has available IPs.

// MetaAllocator implements current allocator interface using
// ServiceCIDR and IPAddress API objects.
type MetaAllocator struct {
	client            networkingv1alpha1client.NetworkingV1alpha1Interface
	serviceCIDRLister networkingv1alpha1listers.ServiceCIDRLister
	serviceCIDRSynced cache.InformerSynced
	ipAddressLister   networkingv1alpha1listers.IPAddressLister
	ipAddressSynced   cache.InformerSynced
	ipAddressInformer networkingv1alpha1informers.IPAddressInformer
	queue             workqueue.TypedRateLimitingInterface[string]

	internalStopCh chan struct{}

	// allocators is a map indexed by the network prefix
	// Multiple ServiceCIDR can contain the same network prefix
	// so we need to store the references from each allocators to
	// the corresponding ServiceCIDRs
	mu         sync.Mutex
	allocators map[string]*item

	ipFamily api.IPFamily
	metrics  bool // enable the metrics collection

	// TODO(aojea): remove with the feature gate DisableAllocatorDualWrite
	bitmapAllocator Interface
}

type item struct {
	allocator    *Allocator
	serviceCIDRs sets.Set[string] // reference of the serviceCIDRs using this Allocator
}

var _ Interface = &MetaAllocator{}

// NewMetaAllocator returns an IP allocator that use the IPAddress
// and ServiceCIDR objects to track the assigned IP addresses,
// using an informer cache as storage.
func NewMetaAllocator(
	client networkingv1alpha1client.NetworkingV1alpha1Interface,
	serviceCIDRInformer networkingv1alpha1informers.ServiceCIDRInformer,
	ipAddressInformer networkingv1alpha1informers.IPAddressInformer,
	isIPv6 bool,
	bitmapAllocator Interface,
) (*MetaAllocator, error) {

	c := newMetaAllocator(client, serviceCIDRInformer, ipAddressInformer, isIPv6, bitmapAllocator)
	go c.run()
	return c, nil
}

// newMetaAllocator is used to build the allocator for testing
func newMetaAllocator(client networkingv1alpha1client.NetworkingV1alpha1Interface,
	serviceCIDRInformer networkingv1alpha1informers.ServiceCIDRInformer,
	ipAddressInformer networkingv1alpha1informers.IPAddressInformer,
	isIPv6 bool,
	bitmapAllocator Interface,
) *MetaAllocator {
	// TODO: make the NewMetaAllocator agnostic of the IP family
	family := api.IPv4Protocol
	if isIPv6 {
		family = api.IPv6Protocol
	}

	c := &MetaAllocator{
		client:            client,
		serviceCIDRLister: serviceCIDRInformer.Lister(),
		serviceCIDRSynced: serviceCIDRInformer.Informer().HasSynced,
		ipAddressLister:   ipAddressInformer.Lister(),
		ipAddressSynced:   ipAddressInformer.Informer().HasSynced,
		ipAddressInformer: ipAddressInformer,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: ControllerName},
		),
		internalStopCh:  make(chan struct{}),
		allocators:      make(map[string]*item),
		ipFamily:        family,
		metrics:         false,
		bitmapAllocator: bitmapAllocator,
	}

	_, _ = serviceCIDRInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: c.enqueServiceCIDR,
		UpdateFunc: func(old, new interface{}) {
			c.enqueServiceCIDR(new)
		},
		// Process the deletion directly in the handler to be able to use the object fields
		// without having to cache them. ServiceCIDRs are protected by finalizers
		// so the "started deletion" logic will be handled in the reconcile loop.
		DeleteFunc: c.deleteServiceCIDR,
	})

	return c
}

func (c *MetaAllocator) enqueServiceCIDR(obj interface{}) {
	key, err := cache.MetaNamespaceKeyFunc(obj)
	if err == nil {
		c.queue.Add(key)
	}
}

func (c *MetaAllocator) deleteServiceCIDR(obj interface{}) {
	serviceCIDR, ok := obj.(*networkingv1alpha1.ServiceCIDR)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			return
		}
		serviceCIDR, ok = tombstone.Obj.(*networkingv1alpha1.ServiceCIDR)
		if !ok {
			return
		}
	}
	klog.Infof("deleting ClusterIP allocator for Service CIDR %v", serviceCIDR)

	c.mu.Lock()
	defer c.mu.Unlock()
	for _, cidr := range serviceCIDR.Spec.CIDRs {
		// skip IP families not supported by this MetaAllocator
		if c.ipFamily != api.IPFamily(convertToV1IPFamily(netutils.IPFamilyOfCIDRString(cidr))) {
			continue
		}
		// get the Allocator used by this ServiceCIDR
		v, ok := c.allocators[cidr]
		if !ok {
			continue
		}
		// remove the reference to this ServiceCIDR
		v.serviceCIDRs.Delete(serviceCIDR.Name)
		if v.serviceCIDRs.Len() > 0 {
			klog.V(2).Infof("deleted Service CIDR from allocator %s, remaining %v", cidr, v.serviceCIDRs)
		} else {
			// if there are no references to this Allocator
			// destroy and remove it from the map
			v.allocator.Destroy()
			delete(c.allocators, cidr)
			klog.Infof("deleted ClusterIP allocator for Service CIDR %s", cidr)
		}
	}
}

func (c *MetaAllocator) run() {
	defer runtime.HandleCrash()
	defer c.queue.ShutDown()
	klog.Info("starting ServiceCIDR Allocator Controller")
	defer klog.Info("stopping ServiceCIDR Allocator Controller")

	// Wait for all involved caches to be synced, before processing items from the queue is started
	if !cache.WaitForCacheSync(c.internalStopCh, c.serviceCIDRSynced, c.ipAddressSynced) {
		runtime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
		return
	}

	// this is single threaded only one serviceCIDR at a time
	go wait.Until(c.runWorker, time.Second, c.internalStopCh)

	<-c.internalStopCh
}

func (c *MetaAllocator) runWorker() {
	for c.processNextItem() {
	}
}

func (c *MetaAllocator) processNextItem() bool {
	// Wait until there is a new item in the working queue
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)
	err := c.syncAllocators()
	// Handle the error if something went wrong during the execution of the business logic
	if err != nil {
		if c.queue.NumRequeues(key) < 5 {
			klog.Infof("error syncing cidr %v: %v", key, err)
			c.queue.AddRateLimited(key)
			return true
		}
	}
	c.queue.Forget(key)
	return true
}

// syncAllocators adds new allocators and syncs the ready state of the allocators
// deletion of allocators is handled directly on the event handler.
func (c *MetaAllocator) syncAllocators() error {
	start := time.Now()
	klog.V(2).Info("syncing ServiceCIDR allocators")
	defer func() {
		klog.V(2).Infof("syncing ServiceCIDR allocators took: %v", time.Since(start))
	}()

	c.mu.Lock()
	defer c.mu.Unlock()

	serviceCIDRs, err := c.serviceCIDRLister.List(labels.Everything())
	if err != nil {
		return err
	}

	for _, serviceCIDR := range serviceCIDRs {
		for _, cidr := range serviceCIDR.Spec.CIDRs {
			// skip IP families not supported by this MetaAllocator
			if c.ipFamily != api.IPFamily(convertToV1IPFamily(netutils.IPFamilyOfCIDRString(cidr))) {
				continue
			}
			// the readiness state of an allocator is an OR of all readiness states
			ready := true
			if !isReady(serviceCIDR) || !serviceCIDR.DeletionTimestamp.IsZero() {
				ready = false
			}

			// check if an allocator already exist for this CIDR
			v, ok := c.allocators[cidr]
			// Update allocator with ServiceCIDR
			if ok {
				v.serviceCIDRs.Insert(serviceCIDR.Name)
				// an Allocator is ready if at least one of the ServiceCIDRs is ready
				if ready {
					v.allocator.ready.Store(true)
				} else if v.serviceCIDRs.Has(serviceCIDR.Name) && len(v.serviceCIDRs) == 1 {
					v.allocator.ready.Store(false)
				}
				klog.Infof("updated ClusterIP allocator for Service CIDR %s", cidr)
				continue
			}

			// Create new allocator for ServiceCIDR
			_, ipnet, err := netutils.ParseCIDRSloppy(cidr) // this was already validated
			if err != nil {
				klog.Infof("error parsing cidr %s", cidr)
				continue
			}
			// New ServiceCIDR, create new allocator
			allocator, err := NewIPAllocator(ipnet, c.client, c.ipAddressInformer)
			if err != nil {
				klog.Infof("error creating new IPAllocator for Service CIDR %s", cidr)
				continue
			}
			if c.metrics {
				allocator.EnableMetrics()
			}
			allocator.ready.Store(ready)
			c.allocators[cidr] = &item{
				allocator:    allocator,
				serviceCIDRs: sets.New[string](serviceCIDR.Name),
			}
			klog.Infof("created ClusterIP allocator for Service CIDR %s", cidr)
		}
	}
	return nil
}

// getAllocator returns any allocator that contains the IP passed as argument.
// if ready is set only an allocator that is ready is returned.
// Allocate operations can work with ANY allocator that is ready, the allocators
// contain references to the IP addresses hence does not matter what allocators have
// the IP. Release operations need to work with ANY allocator independent of its state.
func (c *MetaAllocator) getAllocator(ip net.IP, ready bool) (*Allocator, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	address := servicecidr.IPToAddr(ip)
	// use the first allocator that contains the address
	for cidr, item := range c.allocators {
		prefix, err := netip.ParsePrefix(cidr)
		if err != nil {
			return nil, err
		}
		if servicecidr.PrefixContainsIP(prefix, address) {
			if !ready {
				return item.allocator, nil
			}
			if item.allocator.ready.Load() {
				return item.allocator, nil
			}
		}
	}
	klog.V(2).Infof("Could not get allocator for IP %s", ip.String())
	return nil, ErrMismatchedNetwork
}

func (c *MetaAllocator) AllocateService(service *api.Service, ip net.IP) error {
	allocator, err := c.getAllocator(ip, true)
	if err != nil {
		return err
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.DisableAllocatorDualWrite) {
		cidr := c.bitmapAllocator.CIDR()
		if cidr.Contains(ip) {
			err := c.bitmapAllocator.Allocate(ip)
			if err != nil {
				return err
			}
		}
	}
	return allocator.AllocateService(service, ip)
}

// Allocate attempts to reserve the provided IP. ErrNotInRange or
// ErrAllocated will be returned if the IP is not valid for this range
// or has already been reserved.  ErrFull will be returned if there
// are no addresses left.
// Only for testing, it will fail to create the IPAddress object because
// the Service reference is required.s
func (c *MetaAllocator) Allocate(ip net.IP) error {
	return c.AllocateService(nil, ip)

}

func (c *MetaAllocator) AllocateNextService(service *api.Service) (net.IP, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	// TODO(aojea) add strategy to return a random allocator but
	// taking into consideration the number of addresses of each allocator.
	// Per example, if we have allocator A and B with 256 and 1024 possible
	// addresses each, the chances to get B has to be 4 times the chances to
	// get A so we can spread the load of IPs randomly.
	// However, we need to validate the best strategy before going to Beta.
	isIPv6 := c.ipFamily == api.IPFamily(v1.IPv6Protocol)
	for cidr, item := range c.allocators {
		if netutils.IsIPv6CIDRString(cidr) != isIPv6 {
			continue
		}
		ip, err := item.allocator.AllocateNextService(service)
		if err == nil {
			if !utilfeature.DefaultFeatureGate.Enabled(features.DisableAllocatorDualWrite) {
				cidr := c.bitmapAllocator.CIDR()
				if cidr.Contains(ip) {
					err := c.bitmapAllocator.Allocate(ip)
					if err != nil {
						continue
					}
				}
			}
			return ip, nil
		}
	}
	return nil, ErrFull
}

// AllocateNext return an IP address that wasn't allocated yet.
// Only for testing, it will fail to create the IPAddress object because
// the Service reference is required
func (c *MetaAllocator) AllocateNext() (net.IP, error) {
	return c.AllocateNextService(nil)
}

func (c *MetaAllocator) Release(ip net.IP) error {
	allocator, err := c.getAllocator(ip, false)
	if err != nil {
		return err
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.DisableAllocatorDualWrite) {
		cidr := c.bitmapAllocator.CIDR()
		if cidr.Contains(ip) {
			_ = c.bitmapAllocator.Release(ip)
		}
	}
	return allocator.Release(ip)

}
func (c *MetaAllocator) ForEach(f func(ip net.IP)) {
	ipLabelSelector := labels.Set(map[string]string{
		networkingv1alpha1.LabelIPAddressFamily: string(c.IPFamily()),
		networkingv1alpha1.LabelManagedBy:       ControllerName,
	}).AsSelectorPreValidated()
	ips, err := c.ipAddressLister.List(ipLabelSelector)
	if err != nil {
		return
	}
	for _, ip := range ips {
		f(netutils.ParseIPSloppy(ip.Name))
	}
}

func (c *MetaAllocator) CIDR() net.IPNet {
	return net.IPNet{}

}
func (c *MetaAllocator) IPFamily() api.IPFamily {
	return c.ipFamily
}
func (c *MetaAllocator) Has(ip net.IP) bool {
	allocator, err := c.getAllocator(ip, true)
	if err != nil {
		return false
	}
	return allocator.Has(ip)
}
func (c *MetaAllocator) Destroy() {
	select {
	case <-c.internalStopCh:
	default:
		if !utilfeature.DefaultFeatureGate.Enabled(features.DisableAllocatorDualWrite) {
			c.bitmapAllocator.Destroy()
		}
		close(c.internalStopCh)
	}
}

// for testing
func (c *MetaAllocator) Used() int {
	ipLabelSelector := labels.Set(map[string]string{
		networkingv1alpha1.LabelIPAddressFamily: string(c.IPFamily()),
		networkingv1alpha1.LabelManagedBy:       ControllerName,
	}).AsSelectorPreValidated()
	ips, err := c.ipAddressLister.List(ipLabelSelector)
	if err != nil {
		return 0
	}
	return len(ips)
}

// for testing
func (c *MetaAllocator) Free() int {
	c.mu.Lock()
	defer c.mu.Unlock()

	size := 0
	prefixes := []netip.Prefix{}
	// Get all the existing prefixes
	for cidr := range c.allocators {
		prefix, err := netip.ParsePrefix(cidr)
		if err != nil {
			continue
		}
		prefixes = append(prefixes, prefix)
	}
	// only count the top level prefixes to not double count
	for _, prefix := range prefixes {
		if !isNotContained(prefix, prefixes) {
			continue
		}
		v, ok := c.allocators[prefix.String()]
		if !ok {
			continue
		}
		size += int(v.allocator.size)
	}
	return size - c.Used()
}

func (c *MetaAllocator) EnableMetrics() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.metrics = true
	for _, item := range c.allocators {
		item.allocator.EnableMetrics()
	}
}

// DryRun returns a random allocator
func (c *MetaAllocator) DryRun() Interface {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, item := range c.allocators {
		return item.allocator.DryRun()
	}
	return &Allocator{}
}

func isReady(serviceCIDR *networkingv1alpha1.ServiceCIDR) bool {
	if serviceCIDR == nil {
		return false
	}

	for _, condition := range serviceCIDR.Status.Conditions {
		if condition.Type == networkingv1alpha1.ServiceCIDRConditionReady {
			return condition.Status == metav1.ConditionStatus(metav1.ConditionTrue)
		}
	}
	// assume the ServiceCIDR is Ready, in order to handle scenarios where kcm is not running
	return true
}

// Convert netutils.IPFamily to v1.IPFamily
// TODO: consolidate helpers
// copied from pkg/proxy/util/utils.go
func convertToV1IPFamily(ipFamily netutils.IPFamily) v1.IPFamily {
	switch ipFamily {
	case netutils.IPv4:
		return v1.IPv4Protocol
	case netutils.IPv6:
		return v1.IPv6Protocol
	}

	return v1.IPFamilyUnknown
}

// isNotContained returns true if the prefix is not contained in any
// of the passed prefixes.
func isNotContained(prefix netip.Prefix, prefixes []netip.Prefix) bool {
	for _, p := range prefixes {
		// skip same prefix
		if prefix == p {
			continue
		}
		// 192.168.0.0/24 is contained within 192.168.0.0/16
		if prefix.Overlaps(p) && prefix.Bits() >= p.Bits() {
			return false
		}
	}
	return true
}
