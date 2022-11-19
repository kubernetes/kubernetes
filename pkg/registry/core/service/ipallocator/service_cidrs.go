/*
Copyright 2022 The Kubernetes Authors.

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
	"context"
	"fmt"
	"math"
	"math/rand"
	"net"
	"net/netip"
	"sync"
	"sync/atomic"
	"time"

	networkingv1alpha1 "k8s.io/api/networking/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
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

// Allocator implements current ipallocator interface using API objects
type Allocator struct {
	isIPv6  bool
	rand    *rand.Rand
	startCh chan struct{}
	stopCh  chan struct{}

	client            networkingv1alpha1client.NetworkingV1alpha1Interface
	serviceCIDRLister networkingv1alpha1listers.ServiceCIDRLister
	serviceCIDRSynced cache.InformerSynced
	ipAddressLister   networkingv1alpha1listers.IPAddressLister
	ipAddressSynced   cache.InformerSynced
	queue             workqueue.RateLimitingInterface

	muTree sync.Mutex
	tree   *iptree.Tree

	total uint64 // total number of IPs available in this allocator
}

var _ Interface = &Allocator{}

// NewServiceCIDRAllocator creates a ServiceCIDR IP allocator using the
// configured ranges via the ServiceCIDR API.
func NewServiceCIDRAllocator(
	isIPv6 bool,
	client networkingv1alpha1client.NetworkingV1alpha1Interface,
	serviceCIDRInformer networkingv1alpha1informers.ServiceCIDRInformer,
	ipAddressInformer networkingv1alpha1informers.IPAddressInformer,
) (*Allocator, error) {

	a := &Allocator{
		client:            client,
		isIPv6:            isIPv6,
		rand:              rand.New(rand.NewSource(time.Now().UnixNano())),
		startCh:           make(chan struct{}),
		stopCh:            make(chan struct{}),
		serviceCIDRLister: serviceCIDRInformer.Lister(),
		serviceCIDRSynced: serviceCIDRInformer.Informer().HasSynced,
		ipAddressLister:   ipAddressInformer.Lister(),
		ipAddressSynced:   ipAddressInformer.Informer().HasSynced,
		queue:             workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "ipAllocator"),
		tree:              iptree.New(isIPv6),
	}

	serviceCIDRInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    a.addServiceCIDR,
		UpdateFunc: a.updateServiceCIDR,
		DeleteFunc: a.deleteServiceCIDR,
	})

	go a.run()

	return a, nil
}

func (a *Allocator) addServiceCIDR(obj interface{}) {
	cidr := obj.(*networkingv1alpha1.ServiceCIDR)
	a.queue.Add(cidr.Name)
}

func (a *Allocator) updateServiceCIDR(oldObj, obj interface{}) {
	cidr := obj.(*networkingv1alpha1.ServiceCIDR)
	a.queue.Add(cidr.Name)
}

func (a *Allocator) deleteServiceCIDR(obj interface{}) {
	cidr, ok := obj.(*networkingv1alpha1.ServiceCIDR)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			klog.V(2).Infof("Couldn't get object from tombstone %#v", obj)
			return
		}
		cidr, ok = tombstone.Obj.(*networkingv1alpha1.ServiceCIDR)
		if !ok {
			klog.V(2).Infof("Tombstone contained unexpected object: %#v", obj)
			return
		}
	}
	a.queue.Add(cidr.Name)
}
func (a *Allocator) runWorker() {
	for a.processNextItem() {
	}
}

func (a *Allocator) processNextItem() bool {
	// Wait until there is a new item in the working queue
	key, quit := a.queue.Get()
	if quit {
		return false
	}

	defer a.queue.Done(key)

	err := a.syncTree(key.(string))
	// Handle the error if something went wrong during the execution of the business logic
	if err != nil {
		if a.queue.NumRequeues(key) < 5 {
			klog.Infof("Error syncing cidr %v: %v", key, err)
			a.queue.AddRateLimited(key)
			return true
		}
	}
	a.queue.Forget(key)
	return true
}

// syncTree creates a new IPTree from the current state
// we can not handle the tree using the handlers because there may
// be object with the same CIDR values, and deletes on those CIDRs
// need to consider this problem.
func (a *Allocator) syncTree(key string) error {
	now := time.Now()
	klog.V(0).Infof("Starting sync for CIDR %s", key)
	defer func() {
		klog.V(0).Infof("Finished sync for CIDR %s took %v", key, time.Since(now))
	}()

	newTree := iptree.New(a.isIPv6)
	cidrs, err := a.serviceCIDRLister.List(labels.Everything())
	if err != nil {
		return err
	}
	for _, cidr := range cidrs {
		if cidr.DeletionTimestamp != nil {
			continue
		}
		if a.isIPv6 {
			newTree.Insert(cidr.Spec.IPv6, cidr.Name)
		} else {
			newTree.Insert(cidr.Spec.IPv4, cidr.Name)
		}
	}
	if !a.tree.Equal(newTree) {
		var total int64
		for cidr := range newTree.Parents() {
			prefix := netip.MustParsePrefix(cidr)
			total += rangeSize(prefix)
		}
		atomic.StoreUint64(&a.total, uint64(total))

		a.muTree.Lock()
		a.tree = newTree
		a.muTree.Unlock()
	}
	return nil
}

// Run will not return until stopCh is closed.
func (a *Allocator) run() {
	defer utilruntime.HandleCrash()
	now := time.Now()
	klog.Infof("Starting ServiceCIDRs IP allocator")
	defer klog.Infof("ServiceCIDRs IP allocator ready %v", time.Since(now))

	if !cache.WaitForNamedCacheSync("ipallocator", a.stopCh, a.serviceCIDRSynced, a.ipAddressSynced) {
		return
	}

	// do a first sync
	a.syncTree("bootstrap")
	close(a.startCh)
	// run one worker only
	go wait.Until(a.runWorker, time.Second, a.stopCh)
}

// Allocate the corresponding IP Object
func (a *Allocator) Allocate(ip net.IP) error {
	return a.AllocateService(nil, ip)
}

func (a *Allocator) AllocateService(svc *api.Service, ip net.IP) error {
	select {
	case <-a.startCh:
	default:
		return fmt.Errorf("allocator not ready")
	}
	ctx := context.Background()
	is6 := netutils.IsIPv6(ip)
	if is6 != a.isIPv6 {
		return fmt.Errorf("wrong ip family")
	}
	maskBits := 32
	if is6 {
		maskBits = 128
	}

	// check the IP belongs to a valid range
	a.muTree.Lock()
	cidr, _, ok := a.tree.ShortestPrefixMatch(fmt.Sprintf("%s/%d", ip.String(), maskBits))
	if !ok {
		// no cidr contains the IP
		var msg string
		for k := range a.tree.Parents() {
			msg += k + " "
		}
		a.muTree.Unlock()
		return &ErrNotInRange{ip, msg}
	}
	a.muTree.Unlock()
	prefix, err := netip.ParsePrefix(cidr)
	if err != nil {
		return err
	}
	addr := netip.MustParseAddr(ip.String())
	max := rangeSize(prefix)
	ipFirst := prefix.Masked().Addr().Next()
	ipLast := getIndexedAddr(prefix, max)
	if addr.Less(ipFirst) || addr.Compare(ipLast) == 1 {
		return &ErrNotInRange{ip, cidr}
	}
	ipAddress := networkingv1alpha1.IPAddress{
		ObjectMeta: metav1.ObjectMeta{
			Name: ip.String(),
			Labels: map[string]string{
				networkingv1alpha1.LabelServiceIPAddressFamily: string(a.IPFamily()),
			},
			Finalizers: []string{networkingv1alpha1.IPAddressProtectionFinalizer},
		},
		Spec: networkingv1alpha1.IPAddressSpec{
			ParentRef: serviceToRef(svc),
		},
	}
	_, err = a.client.IPAddresses().Create(ctx, &ipAddress, metav1.CreateOptions{})
	if err != nil {
		return err
	}
	return nil
}

// AllocateNext return an IP address that wasn't allocated yet
func (a *Allocator) AllocateNext() (net.IP, error) {
	return a.AllocateNextService(nil)
}

func (a *Allocator) AllocateNextService(svc *api.Service) (net.IP, error) {
	select {
	case <-a.startCh:
	default:
		return nil, fmt.Errorf("allocator not ready")
	}
	ctx := context.Background()

	// allocate an IP from one of the available ServiceCIDRs
	a.muTree.Lock()
	cidrs := a.tree.Parents()
	a.muTree.Unlock()

	for subnet := range cidrs {
		cidr := netip.MustParsePrefix(subnet)
		max := rangeSize(cidr)

		// TODO this is slow, we should cache the stats per cidr
		// WARNING there can be object with same cidr values
		labelMap := map[string]string{
			networkingv1alpha1.LabelServiceIPAddressFamily: string(a.IPFamily()),
		}
		// get current allocated IPs associated to this range
		ipLabelSelector := labels.Set(labelMap).AsSelectorPreValidated()
		ipAddresses, err := a.ipAddressLister.List(ipLabelSelector)
		if err != nil {
			return nil, err
		}
		// find an empty address within the range
		if int64(len(ipAddresses)) >= max {
			continue
		}

		ipFirst := cidr.Masked().Addr().Next()
		ipLast := getIndexedAddr(cidr, max)
		// start at a random IP within the range
		offset := a.rand.Int63n(max)
		ip := getIndexedAddr(cidr, offset)
		// get some free IP address
		var i int64
		for i = 0; i < max; i++ {
			ip = ip.Next()
			// roll over if it is greater than the last IP allocatable in the range
			if ip.Less(ipFirst) || ip.Compare(ipLast) == 1 {
				ip = ipFirst
			}
			name := ip.String()
			_, err = a.ipAddressLister.Get(name)
			if !apierrors.IsNotFound(err) {
				continue
			}
			ipAddress := networkingv1alpha1.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
					Labels: map[string]string{
						networkingv1alpha1.LabelServiceIPAddressFamily: string(a.IPFamily()),
					},
					Finalizers: []string{networkingv1alpha1.IPAddressProtectionFinalizer},
				},
				Spec: networkingv1alpha1.IPAddressSpec{
					ParentRef: serviceToRef(svc),
				},
			}
			_, err = a.client.IPAddresses().Create(ctx, &ipAddress, metav1.CreateOptions{})
			// this can happen if there is a race and our informer was not updated
			if err == nil {
				return ip.AsSlice(), nil
			}
		}
	}
	return net.IP{}, ErrFull
}

// Release allocated IP
func (a *Allocator) Release(ip net.IP) error {
	select {
	case <-a.startCh:
	default:
		return fmt.Errorf("allocator not ready")
	}
	ctx := context.Background()
	// convert IP to name
	name := ip.String()
	err := a.client.IPAddresses().Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil {
		// backwards compatible
		return nil
	}
	return nil
}

// ForEach executes the function in each allocated IP
func (a *Allocator) ForEach(f func(net.IP)) {
	ipLabelSelector := labels.Set(map[string]string{
		networkingv1alpha1.LabelServiceIPAddressFamily: string(a.IPFamily()),
	}).AsSelectorPreValidated()
	ips, err := a.ipAddressLister.List(ipLabelSelector)
	if err != nil {
		return
	}
	for _, ip := range ips {
		f(netutils.ParseIPSloppy(ip.Name))
	}
}

// backwards compatible
// there are multiple CIDRs now.
func (a *Allocator) CIDR() net.IPNet {
	return net.IPNet{}
}

// for testing
func (a *Allocator) Has(ip net.IP) bool {
	ctx := context.Background()
	// convert IP to name
	name := ip.String()
	ipAddress, err := a.client.IPAddresses().Get(ctx, name, metav1.GetOptions{})
	if err != nil || len(ipAddress.Name) == 0 {
		return false
	}
	return true
}

func (a *Allocator) IPFamily() api.IPFamily {
	if a.isIPv6 {
		return api.IPv6Protocol
	} else {
		return api.IPv4Protocol
	}
}

// for testing
func (a *Allocator) Used() int {
	ipLabelSelector := labels.Set(map[string]string{
		networkingv1alpha1.LabelServiceIPAddressFamily: string(a.IPFamily()),
	}).AsSelectorPreValidated()
	ips, err := a.ipAddressLister.List(ipLabelSelector)
	if err != nil {
		return 0
	}
	return len(ips)
}

// for testing
func (a *Allocator) Free() int {
	return int(atomic.LoadUint64(&a.total)) - a.Used()
}

// Destroy
func (a *Allocator) Destroy() {
	a.startCh = make(chan struct{})
	close(a.stopCh)
}

// DryRun
func (a *Allocator) DryRun() Interface {
	return dryRunAllocator{}
}

// EnableMetrics
func (a *Allocator) EnableMetrics() {
	// TODO
}

type dryRunAllocator struct{}

func (dry dryRunAllocator) Allocate(ip net.IP) error {
	return nil
}

func (dry dryRunAllocator) AllocateNext() (net.IP, error) {
	return net.IP{}, nil
}

func (dry dryRunAllocator) Release(ip net.IP) error {
	return nil
}

func (dry dryRunAllocator) ForEach(cb func(net.IP)) {
}

func (dry dryRunAllocator) CIDR() net.IPNet {
	return net.IPNet{}
}

func (dry dryRunAllocator) IPFamily() api.IPFamily {
	return api.IPv4Protocol
}

func (dry dryRunAllocator) DryRun() Interface {
	return dry
}

func (dry dryRunAllocator) Has(ip net.IP) bool {
	return false
}

func (dry dryRunAllocator) Destroy() {
}

func (dry dryRunAllocator) EnableMetrics() {
}

func rangeSize(cidr netip.Prefix) int64 {
	// subnet was already validated on the API
	mask := 32
	if cidr.Addr().Is6() {
		mask = 128
	}

	bits := cidr.Bits()
	// /32 or /31 for IPv4 and /128 for IPv6 does not have free addresses
	if mask == 32 && bits >= 31 || mask == 128 && bits > 127 {
		return 0
	}
	// this checks that we are not overflowing an int64
	if mask-bits >= 63 {
		return math.MaxInt64
	}
	max := int64(1) << uint(mask-bits)

	// do not consider the base address
	max--
	// do not consider the broadcast address on IPv4
	if cidr.Addr().Is4() {
		max--
	}
	return max
}

// getIndexedAddr returns the address at the provided offset within the subnet
func getIndexedAddr(subnet netip.Prefix, offset int64) netip.Addr {
	base := subnet.Masked().Addr()
	bytes := base.AsSlice()
	if len(bytes) == 4 {
		bytes[0] += byte(offset >> 24)
		bytes[1] += byte(offset >> 16)
		bytes[2] += byte(offset >> 8)
		bytes[3] += byte(offset)
	} else {
		bytes[8] += byte(offset >> 56)
		bytes[9] += byte(offset >> 48)
		bytes[10] += byte(offset >> 40)
		bytes[11] += byte(offset >> 32)
		bytes[12] += byte(offset >> 24)
		bytes[13] += byte(offset >> 16)
		bytes[14] += byte(offset >> 8)
		bytes[15] += byte(offset)
	}

	addr, ok := netip.AddrFromSlice(bytes)
	if !ok {
		panic(bytes)
	}
	if !subnet.Contains(addr) {
		panic(addr)
	}
	return addr

}

// broadcastAddress returns the broadcast address of the subnet
func broadcastAddress(subnet netip.Prefix) netip.Addr {
	base := subnet.Masked().Addr()
	bytes := base.AsSlice()
	n := 8*len(bytes) - subnet.Bits()
	for i := len(bytes) - 1; i >= 0 && n > 0; i-- {
		if n >= 8 {
			bytes[i] = 0xff
			n -= 8
		} else {
			mask := ^uint8(0) >> (8 - n)
			bytes[i] |= mask
			break
		}
	}

	addr, ok := netip.AddrFromSlice(bytes)
	if !ok {
		panic(bytes)
	}
	return addr
}

func serviceToRef(svc *api.Service) *networkingv1alpha1.ParentReference {
	if svc == nil {
		return nil
	}

	return &networkingv1alpha1.ParentReference{
		Group:     "",
		Resource:  "services",
		Namespace: svc.Namespace,
		Name:      svc.Name,
		UID:       svc.UID,
	}
}
