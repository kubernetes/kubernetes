/*
Copyright 2018 The Kubernetes Authors.

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

package net

import (
	"fmt"
	"net"
)

// IPFamily refers to a specific family if not empty, i.e. "4" or "6".
type IPFamily string

// Constants for valid IPFamilys:
const (
	IPFamilyUnknown IPFamily = ""

	IPv4 IPFamily = "4"
	IPv6 IPFamily = "6"
)

// IsDualStackIPs returns true if:
// - all elements of ips are valid
// - at least one IP from each family (v4 and v6) is present
func IsDualStackIPs(ips []net.IP) (bool, error) {
	v4Found := false
	v6Found := false
	for i, ip := range ips {
		switch IPFamilyOf(ip) {
		case IPv4:
			v4Found = true
		case IPv6:
			v6Found = true
		default:
			return false, fmt.Errorf("invalid IP[%d]: %v", i, ip)
		}
	}

	return (v4Found && v6Found), nil
}

// IsDualStackIPStrings returns true if:
// - all elements of ips can be parsed as IPs
// - at least one IP from each family (v4 and v6) is present
func IsDualStackIPStrings(ips []string) (bool, error) {
	parsedIPs := make([]net.IP, 0, len(ips))
	for i, ip := range ips {
		parsedIP := ParseIPSloppy(ip)
		if parsedIP == nil {
			return false, fmt.Errorf("invalid IP[%d]: %v", i, ip)
		}
		parsedIPs = append(parsedIPs, parsedIP)
	}
	return IsDualStackIPs(parsedIPs)
}

// IsDualStackCIDRs returns true if:
// - all elements of cidrs are non-nil
// - at least one CIDR from each family (v4 and v6) is present
func IsDualStackCIDRs(cidrs []*net.IPNet) (bool, error) {
	v4Found := false
	v6Found := false
	for i, cidr := range cidrs {
		switch IPFamilyOfCIDR(cidr) {
		case IPv4:
			v4Found = true
		case IPv6:
			v6Found = true
		default:
			return false, fmt.Errorf("invalid CIDR[%d]: %v", i, cidr)
		}
	}

	return (v4Found && v6Found), nil
}

// IsDualStackCIDRStrings returns if
// - all elements of cidrs can be parsed as CIDRs
// - at least one CIDR from each family (v4 and v6) is present
func IsDualStackCIDRStrings(cidrs []string) (bool, error) {
	parsedCIDRs, err := ParseCIDRs(cidrs)
	if err != nil {
		return false, err
	}
	return IsDualStackCIDRs(parsedCIDRs)
}

// IPFamilyOf returns the IP family of ip, or IPFamilyUnknown if it is invalid.
func IPFamilyOf(ip net.IP) IPFamily {
	switch {
	case ip.To4() != nil:
		return IPv4
	case ip.To16() != nil:
		return IPv6
	default:
		return IPFamilyUnknown
	}
}

// IPFamilyOfString returns the IP family of ip, or IPFamilyUnknown if ip cannot
// be parsed as an IP.
func IPFamilyOfString(ip string) IPFamily {
	return IPFamilyOf(ParseIPSloppy(ip))
}

// IPFamilyOfCIDR returns the IP family of cidr.
func IPFamilyOfCIDR(cidr *net.IPNet) IPFamily {
	if cidr == nil {
		return IPFamilyUnknown
	}
	return IPFamilyOf(cidr.IP)
}

// IPFamilyOfCIDRString returns the IP family of cidr.
func IPFamilyOfCIDRString(cidr string) IPFamily {
	ip, _, _ := ParseCIDRSloppy(cidr)
	return IPFamilyOf(ip)
}

// IsIPv6 returns true if netIP is IPv6 (and false if it is IPv4, nil, or invalid).
func IsIPv6(netIP net.IP) bool {
	return IPFamilyOf(netIP) == IPv6
}

// IsIPv6String returns true if ip contains a single IPv6 address and nothing else. It
// returns false if ip is an empty string, an IPv4 address, or anything else that is not a
// single IPv6 address.
func IsIPv6String(ip string) bool {
	return IPFamilyOfString(ip) == IPv6
}

// IsIPv6CIDR returns true if a cidr is a valid IPv6 CIDR. It returns false if cidr is
// nil or an IPv4 CIDR. Its behavior is not defined if cidr is invalid.
func IsIPv6CIDR(cidr *net.IPNet) bool {
	return IPFamilyOfCIDR(cidr) == IPv6
}

// IsIPv6CIDRString returns true if cidr contains a single IPv6 CIDR and nothing else. It
// returns false if cidr is an empty string, an IPv4 CIDR, or anything else that is not a
// single valid IPv6 CIDR.
func IsIPv6CIDRString(cidr string) bool {
	return IPFamilyOfCIDRString(cidr) == IPv6
}

// IsIPv4 returns true if netIP is IPv4 (and false if it is IPv6, nil, or invalid).
func IsIPv4(netIP net.IP) bool {
	return IPFamilyOf(netIP) == IPv4
}

// IsIPv4String returns true if ip contains a single IPv4 address and nothing else. It
// returns false if ip is an empty string, an IPv6 address, or anything else that is not a
// single IPv4 address.
func IsIPv4String(ip string) bool {
	return IPFamilyOfString(ip) == IPv4
}

// IsIPv4CIDR returns true if cidr is a valid IPv4 CIDR. It returns false if cidr is nil
// or an IPv6 CIDR. Its behavior is not defined if cidr is invalid.
func IsIPv4CIDR(cidr *net.IPNet) bool {
	return IPFamilyOfCIDR(cidr) == IPv4
}

// IsIPv4CIDRString returns true if cidr contains a single IPv4 CIDR and nothing else. It
// returns false if cidr is an empty string, an IPv6 CIDR, or anything else that is not a
// single valid IPv4 CIDR.
func IsIPv4CIDRString(cidr string) bool {
	return IPFamilyOfCIDRString(cidr) == IPv4
}
