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
	"context"
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"net"
	"net/netip"
	"sync/atomic"
	"time"

	networkingv1beta1 "k8s.io/api/networking/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	networkingv1beta1informers "k8s.io/client-go/informers/networking/v1beta1"
	networkingv1beta1client "k8s.io/client-go/kubernetes/typed/networking/v1beta1"
	networkingv1beta1listers "k8s.io/client-go/listers/networking/v1beta1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
	netutils "k8s.io/utils/net"
	utiltrace "k8s.io/utils/trace"
)

const ControllerName = "ipallocator.k8s.io"

// Allocator implements current ipallocator interface using IPAddress API object
// and an informer as backend.
type Allocator struct {
	cidr          *net.IPNet
	prefix        netip.Prefix
	firstAddress  netip.Addr   // first IP address within the range
	offsetAddress netip.Addr   // IP address that delimits the upper and lower subranges
	lastAddress   netip.Addr   // last IP address within the range
	family        api.IPFamily // family is the IP family of this range

	rangeOffset int    // subdivides the assigned IP range to prefer dynamic allocation from the upper range
	size        uint64 // cap the total number of IPs available to maxInt64

	client          networkingv1beta1client.NetworkingV1beta1Interface
	ipAddressLister networkingv1beta1listers.IPAddressLister
	ipAddressSynced cache.InformerSynced
	// ready indicates if the allocator is able to allocate new IP addresses.
	// This is required because it depends on the ServiceCIDR to be ready.
	ready atomic.Bool

	// metrics is a metrics recorder that can be disabled
	metrics     metricsRecorderInterface
	metricLabel string

	rand *rand.Rand
}

var _ Interface = &Allocator{}

// NewIPAllocator returns an IP allocator associated to a network range
// that use the IPAddress objectto track the assigned IP addresses,
// using an informer cache as storage.
func NewIPAllocator(
	cidr *net.IPNet,
	client networkingv1beta1client.NetworkingV1beta1Interface,
	ipAddressInformer networkingv1beta1informers.IPAddressInformer,
) (*Allocator, error) {
	prefix, err := netip.ParsePrefix(cidr.String())
	if err != nil {
		return nil, err
	}

	if prefix.Addr().Is6() && prefix.Bits() < 64 {
		return nil, fmt.Errorf("shortest allowed prefix length for service CIDR is 64, got %d", prefix.Bits())
	}

	// TODO: use the utils/net function once is available
	size := hostsPerNetwork(cidr)
	var family api.IPFamily
	if netutils.IsIPv6CIDR(cidr) {
		family = api.IPv6Protocol
	} else {
		family = api.IPv4Protocol
	}
	// Caching the first, offset and last addresses allows to optimize
	// the search loops by using the netip.Addr iterator instead
	// of having to do conversions with IP addresses.
	// Don't allocate the network's ".0" address.
	ipFirst := prefix.Masked().Addr().Next()
	// Use the broadcast address as last address for IPv6
	ipLast, err := broadcastAddress(prefix)
	if err != nil {
		return nil, err
	}
	// For IPv4 don't use the network's broadcast address
	if family == api.IPv4Protocol {
		ipLast = ipLast.Prev()
	}
	// KEP-3070: Reserve Service IP Ranges For Dynamic and Static IP Allocation
	// calculate the subrange offset
	rangeOffset := calculateRangeOffset(cidr)
	offsetAddress, err := addOffsetAddress(ipFirst, uint64(rangeOffset))
	if err != nil {
		return nil, err
	}
	a := &Allocator{
		cidr:            cidr,
		prefix:          prefix,
		firstAddress:    ipFirst,
		lastAddress:     ipLast,
		rangeOffset:     rangeOffset,
		offsetAddress:   offsetAddress,
		size:            size,
		family:          family,
		client:          client,
		ipAddressLister: ipAddressInformer.Lister(),
		ipAddressSynced: ipAddressInformer.Informer().HasSynced,
		metrics:         &emptyMetricsRecorder{}, // disabled by default
		metricLabel:     cidr.String(),
		rand:            rand.New(rand.NewSource(time.Now().UnixNano())),
	}
	a.ready.Store(true)
	return a, nil
}

func (a *Allocator) createIPAddress(name string, svc *api.Service, scope string) error {
	ipAddress := networkingv1beta1.IPAddress{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				networkingv1beta1.LabelIPAddressFamily: string(a.IPFamily()),
				networkingv1beta1.LabelManagedBy:       ControllerName,
			},
		},
		Spec: networkingv1beta1.IPAddressSpec{
			ParentRef: serviceToRef(svc),
		},
	}
	_, err := a.client.IPAddresses().Create(context.Background(), &ipAddress, metav1.CreateOptions{})
	if err != nil {
		// update metrics
		a.metrics.incrementAllocationErrors(a.metricLabel, scope)
		if apierrors.IsAlreadyExists(err) {
			return ErrAllocated
		}
		return err
	}
	// update metrics
	a.metrics.incrementAllocations(a.metricLabel, scope)
	a.metrics.setAllocated(a.metricLabel, a.Used())
	a.metrics.setAvailable(a.metricLabel, a.Free())
	return nil
}

// Allocate attempts to reserve the provided IP. ErrNotInRange or
// ErrAllocated will be returned if the IP is not valid for this range
// or has already been reserved.  ErrFull will be returned if there
// are no addresses left.
// Only for testing, it will fail to create the IPAddress object because
// the Service reference is required.
func (a *Allocator) Allocate(ip net.IP) error {
	return a.AllocateService(nil, ip)
}

// AllocateService attempts to reserve the provided IP. ErrNotInRange or
// ErrAllocated will be returned if the IP is not valid for this range
// or has already been reserved.  ErrFull will be returned if there
// are no addresses left.
func (a *Allocator) AllocateService(svc *api.Service, ip net.IP) error {
	return a.allocateService(svc, ip, dryRunFalse)
}

func (a *Allocator) allocateService(svc *api.Service, ip net.IP, dryRun bool) error {
	if !a.ready.Load() || !a.ipAddressSynced() {
		return ErrNotReady
	}
	addr, err := netip.ParseAddr(ip.String())
	if err != nil {
		return err
	}

	// check address is within the range of available addresses
	if addr.Less(a.firstAddress) || // requested address is lower than the first address in the subnet
		a.lastAddress.Less(addr) { // the last address in the subnet is lower than the requested address
		if !dryRun {
			// update metrics
			a.metrics.incrementAllocationErrors(a.metricLabel, "static")
		}
		return &ErrNotInRange{ip, a.prefix.String()}
	}
	if dryRun {
		return nil
	}
	start := time.Now()
	err = a.createIPAddress(ip.String(), svc, "static")
	if err != nil {
		return err
	}
	a.metrics.setLatency(a.metricLabel, time.Since(start))
	return nil
}

// AllocateNext return an IP address that wasn't allocated yet.
// Only for testing, it will fail to create the IPAddress object because
// the Service reference is required.
func (a *Allocator) AllocateNext() (net.IP, error) {
	return a.AllocateNextService(nil)
}

// AllocateNext return an IP address that wasn't allocated yet.
func (a *Allocator) AllocateNextService(svc *api.Service) (net.IP, error) {
	return a.allocateNextService(svc, dryRunFalse)
}

// allocateNextService tries to allocate a free IP address within the subnet.
// If the subnet is big enough, it partitions the subnet into two subranges,
// delimited by a.rangeOffset.
// It tries to allocate a free IP address from the upper subnet first and
// falls back to the lower subnet.
// It starts allocating from a random IP within each range.
func (a *Allocator) allocateNextService(svc *api.Service, dryRun bool) (net.IP, error) {
	if !a.ready.Load() || !a.ipAddressSynced() {
		return nil, ErrNotReady
	}
	if dryRun {
		// Don't bother finding a free value. It's racy and not worth the
		// effort to plumb any further.
		return a.CIDR().IP, nil
	}

	trace := utiltrace.New("allocate dynamic ClusterIP address")
	defer trace.LogIfLong(500 * time.Millisecond)
	start := time.Now()

	// rand.Int63n panics for n <= 0 so we need to avoid problems when
	// converting from uint64 to int64
	rangeSize := a.size - uint64(a.rangeOffset)
	var offset uint64
	switch {
	case rangeSize >= math.MaxInt64:
		offset = rand.Uint64()
	case rangeSize == 0:
		return net.IP{}, ErrFull
	default:
		offset = uint64(a.rand.Int63n(int64(rangeSize)))
	}
	iterator := ipIterator(a.offsetAddress, a.lastAddress, offset)
	ip, err := a.allocateFromRange(iterator, svc)
	if err == nil {
		a.metrics.setLatency(a.metricLabel, time.Since(start))
		return ip, nil
	}
	// check the lower range
	if a.rangeOffset != 0 {
		offset = uint64(a.rand.Intn(a.rangeOffset))
		iterator = ipIterator(a.firstAddress, a.offsetAddress.Prev(), offset)
		ip, err = a.allocateFromRange(iterator, svc)
		if err == nil {
			a.metrics.setLatency(a.metricLabel, time.Since(start))
			return ip, nil
		}
	}
	// update metrics
	a.metrics.incrementAllocationErrors(a.metricLabel, "dynamic")
	return net.IP{}, ErrFull
}

// IP iterator allows to iterate over all the IP addresses
// in a range defined by the start and last address.
// It starts iterating at the address position defined by the offset.
// It returns an invalid address to indicate it hasfinished.
func ipIterator(first netip.Addr, last netip.Addr, offset uint64) func() netip.Addr {
	// There are no modulo operations for IP addresses
	modulo := func(addr netip.Addr) netip.Addr {
		if addr.Compare(last) == 1 {
			return first
		}
		return addr
	}
	next := func(addr netip.Addr) netip.Addr {
		return modulo(addr.Next())
	}
	start, err := addOffsetAddress(first, offset)
	if err != nil {
		return func() netip.Addr { return netip.Addr{} }
	}
	start = modulo(start)
	ip := start
	seen := false
	return func() netip.Addr {
		value := ip
		// is the last or the first iteration
		if value == start {
			if seen {
				return netip.Addr{}
			}
			seen = true
		}
		ip = next(ip)
		return value
	}

}

// allocateFromRange allocates an empty IP address from the range of
// IPs between the first and last address (both included), starting
// from the start address.
// TODO: this is a linear search, it can be optimized.
func (a *Allocator) allocateFromRange(iterator func() netip.Addr, svc *api.Service) (net.IP, error) {
	for {
		ip := iterator()
		if !ip.IsValid() {
			break
		}
		name := ip.String()
		_, err := a.ipAddressLister.Get(name)
		// continue if ip already exist
		if err == nil {
			continue
		}
		if !apierrors.IsNotFound(err) {
			klog.Infof("unexpected error: %v", err)
			continue
		}
		// address is not present on the cache, try to allocate it
		err = a.createIPAddress(name, svc, "dynamic")
		// an error can happen if there is a race and our informer was not updated
		// swallow the error and try with the next IP address
		if err != nil {
			klog.Infof("can not create IPAddress %s: %v", name, err)
			continue
		}
		return ip.AsSlice(), nil
	}
	return net.IP{}, ErrFull
}

// Release releases the IP back to the pool. Releasing an
// unallocated IP or an IP out of the range is a no-op and
// returns no error.
func (a *Allocator) Release(ip net.IP) error {
	return a.release(ip, dryRunFalse)
}

func (a *Allocator) release(ip net.IP, dryRun bool) error {
	if dryRun {
		return nil
	}
	name := ip.String()
	// Try to Delete the IPAddress independently of the cache state.
	// The error is ignored for compatibility reasons.
	err := a.client.IPAddresses().Delete(context.Background(), name, metav1.DeleteOptions{})
	if err == nil {
		// update metrics
		a.metrics.setAllocated(a.metricLabel, a.Used())
		a.metrics.setAvailable(a.metricLabel, a.Free())
		return nil
	}
	klog.Infof("error releasing ip %s : %v", name, err)
	return nil
}

// ForEach executes the function on each allocated IP
// This is required to satisfy the Allocator Interface only
func (a *Allocator) ForEach(f func(net.IP)) {
	ipLabelSelector := labels.Set(map[string]string{
		networkingv1beta1.LabelIPAddressFamily: string(a.IPFamily()),
		networkingv1beta1.LabelManagedBy:       ControllerName,
	}).AsSelectorPreValidated()
	ips, err := a.ipAddressLister.List(ipLabelSelector)
	if err != nil {
		return
	}
	for _, ip := range ips {
		f(netutils.ParseIPSloppy(ip.Name))
	}
}

func (a *Allocator) CIDR() net.IPNet {
	return *a.cidr
}

// for testing
func (a *Allocator) Has(ip net.IP) bool {
	// convert IP to name
	name := ip.String()
	ipAddress, err := a.client.IPAddresses().Get(context.Background(), name, metav1.GetOptions{})
	if err != nil || len(ipAddress.Name) == 0 {
		return false
	}
	return true
}

func (a *Allocator) IPFamily() api.IPFamily {
	return a.family
}

// for testing, it assumes this is the allocator is unique for the ipFamily
func (a *Allocator) Used() int {
	ipLabelSelector := labels.Set(map[string]string{
		networkingv1beta1.LabelIPAddressFamily: string(a.IPFamily()),
		networkingv1beta1.LabelManagedBy:       ControllerName,
	}).AsSelectorPreValidated()
	ips, err := a.ipAddressLister.List(ipLabelSelector)
	if err != nil {
		return 0
	}
	return len(ips)
}

// for testing, it assumes this is the allocator is unique for the ipFamily
func (a *Allocator) Free() int {
	return int(a.size) - a.Used()
}

// Destroy
func (a *Allocator) Destroy() {
}

// DryRun
func (a *Allocator) DryRun() Interface {
	return dryRunAllocator{a}
}

// EnableMetrics
func (a *Allocator) EnableMetrics() {
	registerMetrics()
	a.metrics = &metricsRecorder{}
}

// dryRunRange is a shim to satisfy Interface without persisting state.
type dryRunAllocator struct {
	real *Allocator
}

func (dry dryRunAllocator) Allocate(ip net.IP) error {
	return dry.real.allocateService(nil, ip, dryRunTrue)

}

func (dry dryRunAllocator) AllocateNext() (net.IP, error) {
	return dry.real.allocateNextService(nil, dryRunTrue)
}

func (dry dryRunAllocator) Release(ip net.IP) error {
	return dry.real.release(ip, dryRunTrue)
}

func (dry dryRunAllocator) ForEach(cb func(net.IP)) {
	dry.real.ForEach(cb)
}

func (dry dryRunAllocator) CIDR() net.IPNet {
	return dry.real.CIDR()
}

func (dry dryRunAllocator) IPFamily() api.IPFamily {
	return dry.real.IPFamily()
}

func (dry dryRunAllocator) DryRun() Interface {
	return dry
}

func (dry dryRunAllocator) Has(ip net.IP) bool {
	return dry.real.Has(ip)
}

func (dry dryRunAllocator) Destroy() {
}

func (dry dryRunAllocator) EnableMetrics() {
}

// addOffsetAddress returns the address at the provided offset within the subnet
// TODO: move it to k8s.io/utils/net, this is the same as current AddIPOffset()
// but using netip.Addr instead of net.IP
func addOffsetAddress(address netip.Addr, offset uint64) (netip.Addr, error) {
	addressBytes := address.AsSlice()
	addressBig := big.NewInt(0).SetBytes(addressBytes)
	r := big.NewInt(0).Add(addressBig, big.NewInt(int64(offset))).Bytes()
	// r must be 4 or 16 bytes depending of the ip family
	// bigInt conversion to bytes will not take this into consideration
	// and drop the leading zeros, so we have to take this into account.
	lenDiff := len(addressBytes) - len(r)
	if lenDiff > 0 {
		r = append(make([]byte, lenDiff), r...)
	} else if lenDiff < 0 {
		return netip.Addr{}, fmt.Errorf("invalid address %v", r)
	}
	addr, ok := netip.AddrFromSlice(r)
	if !ok {
		return netip.Addr{}, fmt.Errorf("invalid address %v", r)
	}
	return addr, nil
}

// hostsPerNetwork returns the number of available hosts in a subnet.
// The max number is limited by the size of an uint64.
// Number of hosts is calculated with the formula:
// IPv4: 2^x – 2, not consider network and broadcast address
// IPv6: 2^x - 1, not consider network address
// where x is the number of host bits in the subnet.
func hostsPerNetwork(subnet *net.IPNet) uint64 {
	ones, bits := subnet.Mask.Size()
	// this checks that we are not overflowing an int64
	if bits-ones >= 64 {
		return math.MaxUint64
	}
	max := uint64(1) << uint(bits-ones)
	// Don't use the network's ".0" address,
	if max == 0 {
		return 0
	}
	max--
	if netutils.IsIPv4CIDR(subnet) {
		// Don't use the IPv4 network's broadcast address
		if max == 0 {
			return 0
		}
		max--
	}
	return max
}

// broadcastAddress returns the broadcast address of the subnet
// The broadcast address is obtained by setting all the host bits
// in a subnet to 1.
// network 192.168.0.0/24 : subnet bits 24 host bits 32 - 24 = 8
// broadcast address 192.168.0.255
func broadcastAddress(subnet netip.Prefix) (netip.Addr, error) {
	base := subnet.Masked().Addr()
	bytes := base.AsSlice()
	// get all the host bits from the subnet
	n := 8*len(bytes) - subnet.Bits()
	// set all the host bits to 1
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
		return netip.Addr{}, fmt.Errorf("invalid address %v", bytes)
	}
	return addr, nil
}

// serviceToRef obtain the Service Parent Reference
func serviceToRef(svc *api.Service) *networkingv1beta1.ParentReference {
	if svc == nil {
		return nil
	}

	return &networkingv1beta1.ParentReference{
		Group:     "",
		Resource:  "services",
		Namespace: svc.Namespace,
		Name:      svc.Name,
	}
}
