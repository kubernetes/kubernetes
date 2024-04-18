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

package ip

import (
	"net"
	"net/netip"

	netutils "k8s.io/utils/net"
)

// IPFamily represents the IP Family (IPv4 or IPv6) of an IP address or CIDR block.
// (Note that this type is aliased as v1.IPFamily.)
type IPFamily string

const (
	// IPv4Protocol indicates an IPv4 IP or CIDR. (This is the same value as
	// v1.IPv4Protocol.)
	IPv4Protocol IPFamily = "IPv4"
	// IPv6Protocol indicates an IPv6 IP or CIDR. (This is the same value as
	// v1.IPv6Protocol.)
	IPv6Protocol IPFamily = "IPv6"
	// IPFamilyUnknown indicates an unspecified or invalid IP family. (This is the
	// same value as v1.IPFamilyUnknown.)
	IPFamilyUnknown IPFamily = ""
)

type anyIPAddress interface {
	net.IP | netip.Addr | string
}

type anyCIDRAddress interface {
	*net.IPNet | netip.Prefix | string
}

// IPFamilyOf returns the IP family of ip, or IPFamilyUnknown if ip is invalid.
// IPv6-encoded IPv4 addresses (e.g., "::ffff:1.2.3.4") are considered IPv4. ip can be a
// net.IP, a netip.Addr, or a string. String-form IPs can be in any form that would be
// accepted by ParseIP.
//
// (The return value can be used as a v1.IPFamily without needing to be typecast.)
func IPFamilyOf[T anyIPAddress](ip T) IPFamily {
	switch typedIP := interface{}(ip).(type) {
	case net.IP:
		switch {
		case typedIP.To4() != nil:
			return IPv4Protocol
		case typedIP.To16() != nil:
			return IPv6Protocol
		}
	case netip.Addr:
		switch {
		case typedIP.Is4(), typedIP.Is4In6():
			return IPv4Protocol
		case typedIP.Is6():
			return IPv6Protocol
		}
	case string:
		return IPFamilyOf(netutils.ParseIPSloppy(typedIP))
	}

	return IPFamilyUnknown
}

// IPFamilyOfCIDR returns the IP family of cidr (or IPFamilyUnknown if cidr is invalid).
// cidr can be a *net.IPNet, a netip.Prefix, or a string. String-form CIDRs can be in
// any form that would be accepted by ParseCIDR.
//
// (The return value can be used as a v1.IPFamily without needing to be typecast.)
func IPFamilyOfCIDR[T anyCIDRAddress](cidr T) IPFamily {
	switch typedCIDR := interface{}(cidr).(type) {
	case *net.IPNet:
		if typedCIDR != nil {
			return IPFamilyOf(typedCIDR.IP)
		}
	case netip.Prefix:
		return IPFamilyOf(typedCIDR.Addr())
	case string:
		_, parsed, _ := netutils.ParseCIDRSloppy(typedCIDR)
		return IPFamilyOfCIDR(parsed)
	}

	return IPFamilyUnknown
}

// IsIPv4 returns true if IPFamilyOf(ip) is IPv4 (and false if it is IPv6 or invalid).
func IsIPv4[T anyIPAddress](ip T) bool {
	return IPFamilyOf(ip) == IPv4Protocol
}

// IsIPv4CIDR returns true if IPFamilyOfCIDR(cidr) is IPv4. It returns false if cidr is
// invalid or an IPv6 CIDR.
func IsIPv4CIDR[T anyCIDRAddress](cidr T) bool {
	return IPFamilyOfCIDR(cidr) == IPv4Protocol
}

// IsIPv6 returns true if IPFamilyOf(ip) is IPv6 (and false if it is IPv4 or invalid).
func IsIPv6[T anyIPAddress](ip T) bool {
	return IPFamilyOf(ip) == IPv6Protocol
}

// IsIPv6CIDR returns true if IPFamilyOfCIDR(cidr) is IPv6. It returns false if cidr is
// invalid or an IPv4 CIDR.
func IsIPv6CIDR[T anyCIDRAddress](cidr T) bool {
	return IPFamilyOfCIDR(cidr) == IPv6Protocol
}

// IsDualStackIPs returns true if:
// - all elements of ips are "valid" (i.e., IPFamilyOf() returns either IPv4 or IPv6)
// - at least one IP from each family (IPv4 and IPv6) is present
func IsDualStackIPs[T anyIPAddress](ips []T) bool {
	v4Found := false
	v6Found := false
	for _, ip := range ips {
		switch IPFamilyOf(ip) {
		case IPv4Protocol:
			v4Found = true
		case IPv6Protocol:
			v6Found = true
		default:
			return false
		}
	}

	return (v4Found && v6Found)
}

// IsDualStackIPPair returns true if ips contains exactly 1 IPv4 IP and 1 IPv6 IP (in
// either order).
func IsDualStackIPPair[T anyIPAddress](ips []T) bool {
	return len(ips) == 2 && IsDualStackIPs(ips)
}

// IsDualStackCIDRs returns true if:
// - all elements of cidrs are "valid" (i.e., IPFamilyOfCIDR() returns either IPv4 or IPv6)
// - at least one CIDR from each family (v4 and v6) is present
func IsDualStackCIDRs[T anyCIDRAddress](cidrs []T) bool {
	v4Found := false
	v6Found := false
	for _, cidr := range cidrs {
		switch IPFamilyOfCIDR(cidr) {
		case IPv4Protocol:
			v4Found = true
		case IPv6Protocol:
			v6Found = true
		default:
			return false
		}
	}

	return (v4Found && v6Found)
}

// IsDualStackCIDRPair returns true if cidrs contains exactly 1 IPv4 CIDR and 1 IPv6 CIDR
// (in either order).
func IsDualStackCIDRPair[T anyCIDRAddress](cidrs []T) bool {
	return len(cidrs) == 2 && IsDualStackCIDRs(cidrs)
}

// OtherIPFamily returns the other IP family.
func OtherIPFamily(ipFamily IPFamily) IPFamily {
	switch ipFamily {
	case IPv4Protocol:
		return IPv6Protocol
	case IPv6Protocol:
		return IPv4Protocol
	default:
		return IPFamilyUnknown
	}
}
