/*
Copyright 2024 The Kubernetes Authors.

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

package servicecidr

import (
	"fmt"
	"net"
	"net/netip"

	networkingv1alpha1 "k8s.io/api/networking/v1alpha1"
	"k8s.io/apimachinery/pkg/labels"
	networkinglisters "k8s.io/client-go/listers/networking/v1alpha1"
)

// ContainsIP return the list of ServiceCIDR that contains the IP address passed as argument
func ContainsIP(serviceCIDRLister networkinglisters.ServiceCIDRLister, ip net.IP) []*networkingv1alpha1.ServiceCIDR {
	address := IPToAddr(ip)
	return ContainsAddress(serviceCIDRLister, address)
}

// ContainsAddress return the list of ServiceCIDR that contains the address passed as argument
func ContainsAddress(serviceCIDRLister networkinglisters.ServiceCIDRLister, address netip.Addr) []*networkingv1alpha1.ServiceCIDR {
	result := []*networkingv1alpha1.ServiceCIDR{}
	serviceCIDRList, err := serviceCIDRLister.List(labels.Everything())
	if err != nil {
		return result
	}

	for _, serviceCIDR := range serviceCIDRList {
		for _, cidr := range serviceCIDR.Spec.CIDRs {
			if prefix, err := netip.ParsePrefix(cidr); err == nil { // it can not fail since is already validated
				if prefixContainsIP(prefix, address) {
					result = append(result, serviceCIDR)
				}
			}
		}
	}
	return result
}

// prefixContainsIP returns true if the given IP is contained with the prefix,
// is not the network address and also, if IPv4, is not the broadcast address.
// This is required (rather than just `prefix.Contains(ip)`) because a ServiceCIDR
// covering prefix will not allocate those IPs, so a service with one of those IPs
// can't belong to that ServiceCIDR.
func prefixContainsIP(prefix netip.Prefix, ip netip.Addr) bool {
	// if the IP is the network address is not contained
	if prefix.Masked().Addr() == ip {
		return false
	}
	// the broadcast address is not considered contained for IPv4
	if ip.Is4() {
		ipLast, err := broadcastAddress(prefix)
		if err != nil || ipLast == ip {
			return false
		}
	}
	return prefix.Contains(ip)
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

// IPToAddr converts a net.IP to a netip.Addr
// if the net.IP is not valid it returns an empty netip.Addr{}
func IPToAddr(ip net.IP) netip.Addr {
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
