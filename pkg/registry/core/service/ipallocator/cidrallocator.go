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
	"k8s.io/kubernetes/pkg/util/iptree"
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

	muTree sync.Mutex
	tree   *iptree.Tree[*Allocator]

	ipFamily api.IPFamily
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
		tree:              iptree.New[*Allocator](),
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
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err == nil {
		c.queue.Add(key)
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

	err := c.syncTree()
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

// syncTree syncs the ipTrees from the informer cache
// It deletes or creates allocator and sets the corresponding state
func (c *MetaAllocator) syncTree() error {
	now := time.Now()
	defer func() {
		klog.V(2).Infof("Finished sync for CIDRs took %v", time.Since(now))
	}()

	serviceCIDRs, err := c.serviceCIDRLister.List(labels.Everything())
	if err != nil {
		return err
	}

	cidrsSet := sets.New[string]()
	cidrReady := map[string]bool{}
	for _, serviceCIDR := range serviceCIDRs {
		ready := true
		if !isReady(serviceCIDR) || !serviceCIDR.DeletionTimestamp.IsZero() {
			ready = false
		}

		for _, cidr := range serviceCIDR.Spec.CIDRs {
			if c.ipFamily == api.IPFamily(convertToV1IPFamily(netutils.IPFamilyOfCIDRString(cidr))) {
				cidrsSet.Insert(cidr)
				cidrReady[cidr] = ready
			}
		}
	}

	// obtain the existing allocators and set the existing state
	treeSet := sets.New[string]()
	c.muTree.Lock()
	c.tree.DepthFirstWalk(c.ipFamily == api.IPv6Protocol, func(k netip.Prefix, v *Allocator) bool {
		v.ready.Store(cidrReady[k.String()])
		treeSet.Insert(k.String())
		return false
	})
	c.muTree.Unlock()
	cidrsToRemove := treeSet.Difference(cidrsSet)
	cidrsToAdd := cidrsSet.Difference(treeSet)

	errs := []error{}
	// Add new allocators
	for _, cidr := range cidrsToAdd.UnsortedList() {
		_, ipnet, err := netutils.ParseCIDRSloppy(cidr)
		if err != nil {
			return err
		}
		// New ServiceCIDR, create new allocator
		allocator, err := NewIPAllocator(ipnet, c.client, c.ipAddressInformer)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		allocator.ready.Store(cidrReady[cidr])
		prefix, err := netip.ParsePrefix(cidr)
		if err != nil {
			return err
		}
		c.addAllocator(prefix, allocator)
		klog.Infof("Created ClusterIP allocator for Service CIDR %s", cidr)
	}
	// Remove allocators that no longer exist
	for _, cidr := range cidrsToRemove.UnsortedList() {
		prefix, err := netip.ParsePrefix(cidr)
		if err != nil {
			return err
		}
		c.deleteAllocator(prefix)
	}

	return utilerrors.NewAggregate(errs)
}

func (c *MetaAllocator) getAllocator(ip net.IP) (*Allocator, error) {
	c.muTree.Lock()
	defer c.muTree.Unlock()

	address := ipToAddr(ip)
	prefix := netip.PrefixFrom(address, address.BitLen())
	// Use the largest subnet to allocate addresses because
	// all the other subnets will be contained.
	_, allocator, ok := c.tree.ShortestPrefixMatch(prefix)
	if !ok {
		klog.V(2).Infof("Could not get allocator for IP %s", ip.String())
		return nil, ErrMismatchedNetwork
	}
	return allocator, nil
}

func (c *MetaAllocator) addAllocator(cidr netip.Prefix, allocator *Allocator) {
	c.muTree.Lock()
	defer c.muTree.Unlock()
	c.tree.InsertPrefix(cidr, allocator)
}

func (c *MetaAllocator) deleteAllocator(cidr netip.Prefix) {
	c.muTree.Lock()
	defer c.muTree.Unlock()
	ok := c.tree.DeletePrefix(cidr)
	if ok {
		klog.V(3).Infof("CIDR %s deleted", cidr)
	}
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
	c.muTree.Lock()
	defer c.muTree.Unlock()

	// TODO(aojea) add strategy to return a random allocator but
	// taking into consideration the number of addresses of each allocator.
	// Per example, if we have allocator A and B with 256 and 1024 possible
	// addresses each, the chances to get B has to be 4 times the chances to
	// get A so we can spread the load of IPs randomly.
	// However, we need to validate the best strategy before going to Beta.
	isIPv6 := c.ipFamily == api.IPFamily(v1.IPv6Protocol)
	for _, allocator := range c.tree.TopLevelPrefixes(isIPv6) {
		ip, err := allocator.AllocateNextService(service)
		if err == nil {
			return ip, nil
		}
	}
	return nil, ErrFull
}

func (c *MetaAllocator) AllocateNext() (net.IP, error) {
	c.muTree.Lock()
	defer c.muTree.Unlock()

	// TODO(aojea) add strategy to return a random allocator but
	// taking into consideration the number of addresses of each allocator.
	// Per example, if we have allocator A and B with 256 and 1024 possible
	// addresses each, the chances to get B has to be 4 times the chances to
	// get A so we can spread the load of IPs randomly.
	// However, we need to validate the best strategy before going to Beta.
	isIPv6 := c.ipFamily == api.IPFamily(v1.IPv6Protocol)
	for _, allocator := range c.tree.TopLevelPrefixes(isIPv6) {
		ip, err := allocator.AllocateNext()
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
	c.muTree.Lock()
	defer c.muTree.Unlock()

	size := 0
	isIPv6 := c.ipFamily == api.IPFamily(v1.IPv6Protocol)
	for _, allocator := range c.tree.TopLevelPrefixes(isIPv6) {
		size += int(allocator.size)
	}
	return size - c.Used()
}

func (c *MetaAllocator) EnableMetrics() {}

// DryRun returns a random allocator
func (c *MetaAllocator) DryRun() Interface {
	c.muTree.Lock()
	defer c.muTree.Unlock()
	isIPv6 := c.ipFamily == api.IPFamily(v1.IPv6Protocol)
	for _, allocator := range c.tree.TopLevelPrefixes(isIPv6) {
		return allocator.DryRun()
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

// ipToAddr converts a net.IP to a netip.Addr
// if the net.IP is not valid it returns an empty netip.Addr{}
func ipToAddr(ip net.IP) netip.Addr {
	// https://pkg.go.dev/net/netip#AddrFromSlice can return an IPv4 in IPv6 format
	// so we have to check the IP family to return exactly the format that we want
	// address, _ := netip.AddrFromSlice(net.ParseIPSloppy(192.168.0.1)) returns
	// an address like ::ffff:192.168.0.1/32
	bytes := ip.To4()
	if bytes == nil {
		bytes = ip.To16()
	}
	// AddrFromSlice returns Addr{}, false if the input is invalid.
	address, _ := netip.AddrFromSlice(bytes)
	return address
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
