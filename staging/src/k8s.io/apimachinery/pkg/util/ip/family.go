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

package ip

import (
	"net"
	"net/netip"

	"k8s.io/api/core/v1"
	netutils "k8s.io/utils/net"
)

// IPFamilyOf returns the IP family of ip (or "" if ip is invalid).
func IPFamilyOf(ip netip.Addr) v1.IPFamily {
	switch {
	case ip.Is4(), ip.Is4In6():
		return v1.IPv4Protocol
	case ip.Is6():
		return v1.IPv6Protocol
	default:
		return ""
	}
}

func ipFamilyOfNetIP(ip net.IP) v1.IPFamily {
	switch {
	case ip.To4() != nil:
		return v1.IPv4Protocol
	case ip.To16() != nil:
		return v1.IPv6Protocol
	default:
		return ""
	}
}

// IPFamilyOfString returns the IP family of ipStr, or "" if ipStr cannot be parsed as an
// IP address. ipStr can be a valid or "legacy" IP string.
func IPFamilyOfString(ipStr string) v1.IPFamily {
	return ipFamilyOfNetIP(netutils.ParseIPSloppy(ipStr))
}

// IPFamilyOfCIDR returns the IP family of cidr (or "" if cidr is invalid).
func IPFamilyOfCIDR(cidr netip.Prefix) v1.IPFamily {
	return IPFamilyOf(cidr.Addr())
}

// IPFamilyOfCIDRString returns the IP family of cidrStr, or "" if cidrStr cannot be
// parsed as a CIDR. cidrStr can be a valid or "legacy" CIDR string.
func IPFamilyOfCIDRString(cidr string) v1.IPFamily {
	ip, _, _ := netutils.ParseCIDRSloppy(cidr)
	return ipFamilyOfNetIP(ip)
}

// IsIPv6 returns true if ip is IPv6 (and false if it is IPv4 or invalid).
func IsIPv6(ip netip.Addr) bool {
	return IPFamilyOf(ip) == v1.IPv6Protocol
}

// IsIPv6String returns true if ipStr contains a single IPv6 address (valid or "legacy")
// and nothing else. It returns false if ipStr is an empty string, an IPv4 address, or
// anything else that is not a single IPv6 address.
func IsIPv6String(ipStr string) bool {
	return IPFamilyOfString(ipStr) == v1.IPv6Protocol
}

// IsIPv6CIDR returns true if a cidr is a valid IPv6 CIDR. It returns false if cidr is
// invalid or an IPv4 CIDR.
func IsIPv6CIDR(cidr netip.Prefix) bool {
	return IPFamilyOfCIDR(cidr) == v1.IPv6Protocol
}

// IsIPv6CIDRString returns true if cidrStr contains a single IPv6 CIDR (valid or
// "legacy") and nothing else. It returns false if cidrStr is an empty string, an IPv4
// CIDR, or anything else that is not a single valid IPv6 CIDR.
func IsIPv6CIDRString(cidrStr string) bool {
	return IPFamilyOfCIDRString(cidrStr) == v1.IPv6Protocol
}

// IsIPv4 returns true if ip is IPv4 (and false if it is IPv6 or invalid).
func IsIPv4(ip netip.Addr) bool {
	return IPFamilyOf(ip) == v1.IPv4Protocol
}

// IsIPv4String returns true if ipStr contains a single IPv4 address (valid or "legacy")
// and nothing else. It returns false if ipStr is an empty string, an IPv6 address, or
// anything else that is not a single IPv4 address.
func IsIPv4String(ipStr string) bool {
	return IPFamilyOfString(ipStr) == v1.IPv4Protocol
}

// IsIPv4CIDR returns true if cidr is a valid IPv4 CIDR. It returns false if cidr is
// invalid or an IPv6 CIDR.
func IsIPv4CIDR(cidr netip.Prefix) bool {
	return IPFamilyOfCIDR(cidr) == v1.IPv4Protocol
}

// IsIPv4CIDRString returns true if cidrStr contains a single IPv4 CIDR (valid or
// "legacy") and nothing else. It returns false if cidrStr is an empty string, an IPv6
// CIDR, or anything else that is not a single valid IPv4 CIDR.
func IsIPv4CIDRString(cidrStr string) bool {
	return IPFamilyOfCIDRString(cidrStr) == v1.IPv4Protocol
}

// IsDualStackIPs returns true if:
// - all elements of ips are valid
// - at least one IP from each family (v4 and v6) is present
func IsDualStackIPs(ips []netip.Addr) bool {
	v4Found := false
	v6Found := false
	for _, ip := range ips {
		switch IPFamilyOf(ip) {
		case v1.IPv4Protocol:
			v4Found = true
		case v1.IPv6Protocol:
			v6Found = true
		default:
			return false
		}
	}

	return (v4Found && v6Found)
}

// IsDualStackIPStrings returns true if:
// - all elements of ipStrs can be parsed as IPs (valid or "legacy")
// - at least one IP from each family (v4 and v6) is present
func IsDualStackIPStrings(ipStrs []string) bool {
	v4Found := false
	v6Found := false
	for _, ipStr := range ipStrs {
		switch IPFamilyOfString(ipStr) {
		case v1.IPv4Protocol:
			v4Found = true
		case v1.IPv6Protocol:
			v6Found = true
		default:
			return false
		}
	}

	return (v4Found && v6Found)
}

// IsDualStackCIDRs returns true if:
// - all elements of cidrs are non-nil
// - at least one CIDR from each family (v4 and v6) is present
func IsDualStackCIDRs(cidrs []netip.Prefix) bool {
	v4Found := false
	v6Found := false
	for _, cidr := range cidrs {
		switch IPFamilyOfCIDR(cidr) {
		case v1.IPv4Protocol:
			v4Found = true
		case v1.IPv6Protocol:
			v6Found = true
		default:
			return false
		}
	}

	return (v4Found && v6Found)
}

// IsDualStackCIDRStrings returns if
// - all elements of cidrStrs can be parsed as CIDRs (valid or "legacy")
// - at least one CIDR from each family (v4 and v6) is present
func IsDualStackCIDRStrings(cidrStrs []string) bool {
	v4Found := false
	v6Found := false
	for _, cidrStr := range cidrStrs {
		switch IPFamilyOfCIDRString(cidrStr) {
		case v1.IPv4Protocol:
			v4Found = true
		case v1.IPv6Protocol:
			v6Found = true
		default:
			return false
		}
	}

	return (v4Found && v6Found)
}

// OtherIPFamily returns the other ip family
func OtherIPFamily(ipFamily v1.IPFamily) v1.IPFamily {
	if ipFamily == v1.IPv6Protocol {
		return v1.IPv4Protocol
	}
	return v1.IPv6Protocol
}
