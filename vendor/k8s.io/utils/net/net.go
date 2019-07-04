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

// ParseCIDRs parses a list of cidrs and return error if any is invalid.
// order is maintained
func ParseCIDRs(cidrsString []string) ([]*net.IPNet, error) {
	cidrs := make([]*net.IPNet, 0, len(cidrsString))
	for _, cidrString := range cidrsString {
		_, cidr, err := net.ParseCIDR(cidrString)
		if err != nil {
			return nil, fmt.Errorf("failed to parse cidr value:%q with error:%v", cidrString, err)
		}
		cidrs = append(cidrs, cidr)
	}
	return cidrs, nil
}

// IsDualStackIPs returns if a slice of ips is:
// - all are valid ips
// - at least one ip from each family (v4 or v6)
func IsDualStackIPs(ips []net.IP) (bool, error) {
	v4Found := false
	v6Found := false
	for _, ip := range ips {
		if ip == nil {
			return false, fmt.Errorf("ip %v is invalid", ip)
		}

		if v4Found && v6Found {
			continue
		}

		if IsIPv6(ip) {
			v6Found = true
			continue
		}

		v4Found = true
	}

	return (v4Found && v6Found), nil
}

// IsDualStackIPStrings returns if
// - all are valid ips
// - at least one ip from each family (v4 or v6)
func IsDualStackIPStrings(ips []string) (bool, error) {
	parsedIPs := make([]net.IP, 0, len(ips))
	for _, ip := range ips {
		parsedIP := net.ParseIP(ip)
		parsedIPs = append(parsedIPs, parsedIP)
	}
	return IsDualStackIPs(parsedIPs)
}

// IsDualStackCIDRs returns if
// - all are valid cidrs
// - at least one cidr from each family (v4 or v6)
func IsDualStackCIDRs(cidrs []*net.IPNet) (bool, error) {
	v4Found := false
	v6Found := false
	for _, cidr := range cidrs {
		if cidr == nil {
			return false, fmt.Errorf("cidr %v is invalid", cidr)
		}

		if v4Found && v6Found {
			continue
		}

		if IsIPv6(cidr.IP) {
			v6Found = true
			continue
		}
		v4Found = true
	}

	return v4Found && v6Found, nil
}

// IsDualStackCIDRStrings returns if
// - all are valid cidrs
// - at least one cidr from each family (v4 or v6)
func IsDualStackCIDRStrings(cidrs []string) (bool, error) {
	parsedCIDRs, err := ParseCIDRs(cidrs)
	if err != nil {
		return false, err
	}
	return IsDualStackCIDRs(parsedCIDRs)
}

// IsIPv6 returns if netIP is IPv6.
func IsIPv6(netIP net.IP) bool {
	return netIP != nil && netIP.To4() == nil
}

// IsIPv6String returns if ip is IPv6.
func IsIPv6String(ip string) bool {
	netIP := net.ParseIP(ip)
	return IsIPv6(netIP)
}

// IsIPv6CIDRString returns if cidr is IPv6.
// This assumes cidr is a valid CIDR.
func IsIPv6CIDRString(cidr string) bool {
	ip, _, _ := net.ParseCIDR(cidr)
	return IsIPv6(ip)
}

// IsIPv6CIDR returns if a cidr is ipv6
func IsIPv6CIDR(cidr *net.IPNet) bool {
	ip := cidr.IP
	return IsIPv6(ip)
}
