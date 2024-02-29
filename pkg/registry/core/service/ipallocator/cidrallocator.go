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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	networkingv1alpha1informers "k8s.io/client-go/informers/networking/v1alpha1"
	networkingv1alpha1client "k8s.io/client-go/kubernetes/typed/networking/v1alpha1"
	networkingv1alpha1listers "k8s.io/client-go/listers/networking/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/util/servicecidr"
	netutils "k8s.io/utils/net"
)

// MetaAllocator maintains a Tree with the ServiceCIDRs containing an IP Allocator
// on the nodes. Since each allocator doesn't stored the IPAddresses because it reads
// them from the informer cache, it is cheap to create and delete IP Allocators.
// MetaAllocator forwards the request to any of the internal allocators that has free
// addresses.

// MetaAllocator implements current allocator interface using
// ServiceCIDR and IPAddress API objects.
type MetaAllocator struct {
	client            networkingv1alpha1client.NetworkingV1alpha1Interface
	serviceCIDRLister networkingv1alpha1listers.ServiceCIDRLister
	serviceCIDRSynced cache.InformerSynced
	ipAddressLister   networkingv1alpha1listers.IPAddressLister
	ipAddressSynced   cache.InformerSynced
	ipAddressInformer networkingv1alpha1informers.IPAddressInformer
	queue             workqueue.RateLimitingInterface

	internalStopCh chan struct{}

	// allocators is a map indexed by the network prefix
	// that contains its corresponding Allocator and a
	// reference to the ServiceCIDRs using the Allocator
	mu         sync.Mutex
	allocators map[string]*item

	ipFamily api.IPFamily
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
) (*MetaAllocator, error) {

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
		queue:             workqueue.NewRateLimitingQueueWithConfig(workqueue.DefaultControllerRateLimiter(), workqueue.RateLimitingQueueConfig{Name: ControllerName}),
		internalStopCh:    make(chan struct{}),
		allocators:        make(map[string]*item),
		ipFamily:          family,
	}

	_, _ = serviceCIDRInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addServiceCIDR,
		UpdateFunc: c.updateServiceCIDR,
		DeleteFunc: c.deleteServiceCIDR,
	})

	go c.run()

	return c, nil
}

func (c *MetaAllocator) addServiceCIDR(obj interface{}) {
	key, err := cache.MetaNamespaceKeyFunc(obj)
	if err == nil {
		c.queue.Add(key)
	}
}
func (c *MetaAllocator) updateServiceCIDR(old, new interface{}) {
	key, err := cache.MetaNamespaceKeyFunc(new)
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
	c.mu.Lock()
	defer c.mu.Unlock()
	klog.Infof("Deleted ClusterIP allocator for Service CIDR %v", serviceCIDR)

	for _, cidr := range serviceCIDR.Spec.CIDRs {
		// get the Allocator used by this ServiceCIDR
		v, ok := c.allocators[cidr]
		if !ok {
			continue
		}
		// remove the reference to this ServiceCIDR
		v.serviceCIDRs.Delete(serviceCIDR.Name)
		if v.serviceCIDRs.Len() > 0 {
			klog.V(4).Infof("Deleted Service CIDR from allocator %s %v", cidr, v.serviceCIDRs)
			continue
		}
		// if there are no references to this Allocator
		// destroy and remove it from the map
		v.allocator.Destroy()
		delete(c.allocators, cidr)
		klog.Infof("Deleted ClusterIP allocator for Service CIDR %s", cidr)
	}
}

func (c *MetaAllocator) run() {
	defer runtime.HandleCrash()
	defer c.queue.ShutDown()
	klog.Info("Starting ServiceCIDR Allocator Controller")
	defer klog.Info("Stopping ServiceCIDR Allocator Controllerr")

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

	err := c.syncAllocator(key.(string))
	// Handle the error if something went wrong during the execution of the business logic
	if err != nil {
		if c.queue.NumRequeues(key) < 5 {
			klog.Infof("Error syncing cidr %v: %v", key, err)
			c.queue.AddRateLimited(key)
			return true
		}
	}
	c.queue.Forget(key)
	return true
}

// syncAllocators deletes or creates allocators and sets the corresponding state
func (c *MetaAllocator) syncAllocator(key string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	errs := []error{}
	now := time.Now()
	defer func() {
		klog.V(2).Infof("Finished sync for CIDRs took %v", time.Since(now))
	}()

	serviceCIDR, err := c.serviceCIDRLister.Get(key)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}

	// Add or Update
	ready := true
	if !isReady(serviceCIDR) || !serviceCIDR.DeletionTimestamp.IsZero() {
		ready = false
	}

	for _, cidr := range serviceCIDR.Spec.CIDRs {
		// skip IP families not supported by this MetaAllocator
		if c.ipFamily != api.IPFamily(convertToV1IPFamily(netutils.IPFamilyOfCIDRString(cidr))) {
			continue
		}

		v, ok := c.allocators[cidr]
		// allocator already exist for this CIDR
		// add new ServiceCIDR reference to the set and update the state
		// an Allocator is ready if at least one of the ServiceCIDRs is ready
		if ok {
			if ready {
				v.allocator.ready.Store(true)
				v.serviceCIDRs.Insert(serviceCIDR.Name)
				klog.Infof("Update ClusterIP allocator for Service CIDR %s", cidr)
			}
			continue
		}
		// there is no allocator for this prefix
		_, ipnet, err := netutils.ParseCIDRSloppy(cidr)
		if err != nil {
			errs = append(errs, err)
			continue
		}

		// New ServiceCIDR, create new allocator
		allocator, err := NewIPAllocator(ipnet, c.client, c.ipAddressInformer)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		allocator.ready.Store(ready)
		c.allocators[cidr] = &item{
			allocator:    allocator,
			serviceCIDRs: sets.New[string](serviceCIDR.Name),
		}
		klog.Infof("Created ClusterIP allocator for Service CIDR %s", cidr)

	}

	return utilerrors.NewAggregate(errs)
}

func (c *MetaAllocator) getAllocator(ip net.IP) (*Allocator, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	address := servicecidr.IPToAddr(ip)
	// use the first allocator that contains the address
	for cidr, item := range c.allocators {
		prefix, err := netip.ParsePrefix(cidr)
		if err != nil {
			return nil, err
		}
		if servicecidr.PrefixContainIP(prefix, address) && item.allocator.ready.Load() {
			return item.allocator, nil
		}
	}
	klog.V(2).Infof("Could not get allocator for IP %s", ip.String())
	return nil, ErrMismatchedNetwork
}

func (c *MetaAllocator) AllocateService(service *api.Service, ip net.IP) error {
	allocator, err := c.getAllocator(ip)
	if err != nil {
		return err
	}
	return allocator.AllocateService(service, ip)
}

func (c *MetaAllocator) Allocate(ip net.IP) error {
	allocator, err := c.getAllocator(ip)
	if err != nil {
		return err
	}
	return allocator.Allocate(ip)
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
			return ip, nil
		}
	}
	return nil, ErrFull
}

func (c *MetaAllocator) AllocateNext() (net.IP, error) {
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
		if netutils.IsIPv6String(cidr) != isIPv6 {
			continue
		}
		ip, err := item.allocator.AllocateNext()
		if err == nil {
			return ip, nil
		}
	}
	return nil, ErrFull
}

func (c *MetaAllocator) Release(ip net.IP) error {
	allocator, err := c.getAllocator(ip)
	if err != nil {
		return err
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
	allocator, err := c.getAllocator(ip)
	if err != nil {
		return false
	}
	return allocator.Has(ip)
}
func (c *MetaAllocator) Destroy() {
	select {
	case <-c.internalStopCh:
	default:
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
	// We should not double count so only take into account
	// the prefixes that are not contained.
	prefixes := []netip.Prefix{}
	for cidr := range c.allocators {
		prefix, err := netip.ParsePrefix(cidr)
		if err != nil {
			continue
		}
		prefixes = append(prefixes, prefix)
	}
	for _, prefix := range prefixes {
		if !isSupernet(prefix, prefixes) {
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

// TODO
func (c *MetaAllocator) EnableMetrics() {}

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

// isSupernet returns true if the prefix is not contained in any
// of the passed prefixes.
func isSupernet(prefix netip.Prefix, prefixes []netip.Prefix) bool {
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
