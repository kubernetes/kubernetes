/*
Copyright 2021 The Kubernetes Authors.

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

package allocation

import (
	"context"
	"math/rand"
	"net"

	allocation "k8s.io/api/allocation/v1alpha1"
	allocationclient "k8s.io/client-go/kubernetes/typed/allocation/v1alpha1"
	utilallocation "k8s.io/kubernetes/pkg/apis/allocation/util"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	utilnet "k8s.io/utils/net"
)

// default allocator well-known-names
const (
	DefaultIPv4Allocator = "allocator.ipv4.k8s.io"
	DefaultIPv6Allocator = "allocator.ipv6.k8s.io"
)

// Range implements current ipallocator interface using API objects
type Range struct {
	name   string
	client allocationclient.AllocationV1alpha1Interface
}

var _ ipallocator.Interface = &Range{}

// NewAllocatorCIDRRange creates a Range using a net.IPNet
// TODO what is the right approach, using a client or reststorage?
func NewAllocatorCIDRRange(cidr *net.IPNet, client allocationclient.AllocationV1alpha1Interface) (*Range, error) {
	name := DefaultIPv4Allocator
	if utilnet.IsIPv6CIDR(cidr) {
		name = DefaultIPv6Allocator
	}
	return &Range{
		client: client,
		name:   name,
	}, nil
}

// Allocate the corresponding IP Object
func (r *Range) Allocate(ip net.IP) error {
	ctx := context.Background()
	// convert IP to name
	name := utilallocation.IPToDecimal(ip)
	ipRangeRef := allocation.IPRangeRef{
		APIGroup: "allocation.k8s.io",
		Kind:     "IPRange",
		Name:     r.name,
	}
	ipAddress := allocation.IPAddress{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: allocation.IPAddressSpec{
			Address:    ip.String(),
			IPRangeRef: ipRangeRef,
		},
	}
	// TODO: instead of Create the object Updated the Status to Allocated so we can reuse
	_, err := r.client.IPAddresses().Create(ctx, &ipAddress, metav1.CreateOptions{})
	return err
}

// AllocateNext return an IP address that wasn't allocated yet
func (r *Range) AllocateNext() (net.IP, error) {
	ctx := context.Background()
	// get current CIDR
	cidr := r.CIDR()
	max := utilnet.RangeSize(&cidr)
	// get current allocated IPs
	// TODO should we implement a fieldselector based on the cluster IPRange referenced
	// ipRangeSelector := fields.Set{"spec.ipRangeref": r.name}.AsSelector().String()
	// ipAddresses, err := r.client.IPAddresses().List(ctx, metav1.ListOptions{FieldSelector: ipRangeSelector})
	ipAddresses, err := r.client.IPAddresses().List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	// find an empty address within the range
	if int64(len(ipAddresses.Items)) >= max {
		return net.IP{}, ipallocator.ErrFull
	}
	// create a Set with all the allocated IPs
	// TODO check status to get only the Free ones so we don't have
	// to find for an empty one
	ipset := make(utilnet.IPSet)
	for _, ipAddress := range ipAddresses.Items {
		// ip was already validated
		ip := net.ParseIP(ipAddress.Spec.Address)
		// TODO meanwhile we don't implement a field selector to get only the
		// IPAddresses that belongs to current allocator, we do it here
		if cidr.Contains(ip) {
			ipset.Insert(ip)
		}
	}

	offset := rand.Int63n(max)
	var i int64
	for i = 0; i < max; i++ {
		at := (offset + i) % max
		ip, err := utilnet.GetIndexedIP(&cidr, int(at))
		if err != nil {
			return net.IP{}, ipallocator.ErrAllocated
		}
		if !ipset.Has(ip) {
			err := r.Allocate(ip)
			// TODO it can happen we fail to allocate
			// because it was already allocated by
			// other apiserver
			// if err is already allocated continue
			// otherwise return the error
			if err != nil {
				continue
			}
			return ip, nil
		}
	}

	return net.IP{}, ipallocator.ErrFull
}

// Release allocated IP
func (r *Range) Release(ip net.IP) error {
	ctx := context.Background()
	// convert IP to name
	name := utilallocation.IPToDecimal(ip)
	// TODO: instead of Delete the object Updated the Status to Free so we can reuse
	return r.client.IPAddresses().Delete(ctx, name, metav1.DeleteOptions{})
}

// ForEach executes the function in each allocated IP
func (r *Range) ForEach(func(net.IP)) {
	// it may not be needed with this approach
}

// CIDR return the current range CIDR
func (r *Range) CIDR() net.IPNet {
	ctx := context.Background()
	ipRange, err := r.client.IPRanges().Get(ctx, r.name, metav1.GetOptions{})
	if err != nil {
		klog.Warningf("error obtaining current allocator object %s: %v", r.name, err)
		return net.IPNet{}
	}
	// Range has been validated
	_, subnet, _ := net.ParseCIDR(ipRange.Spec.Range)
	return *subnet
}

// Has is only used for testing
func (r *Range) Has(ip net.IP) bool {
	ctx := context.Background()
	// convert IP to name
	name := utilallocation.IPToDecimal(ip)
	ipAddress, err := r.client.IPAddresses().Get(ctx, name, metav1.GetOptions{})
	if err != nil || len(ipAddress.Name) == 0 {
		return false
	}
	return true
}
