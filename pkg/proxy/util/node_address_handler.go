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

package util

import (
	"fmt"
	"net"

	"k8s.io/api/core/v1"
	netutils "k8s.io/utils/net"
)

// NodeAddressHandler is used to handle NodePortAddresses,
// HealthzBindAddresses and MetricsBindAddresses.
type NodeAddressHandler struct {
	cidrStrings []string

	cidrs                []*net.IPNet
	containsIPv4Loopback bool
	matchAll             bool
}

// RFC 5735 127.0.0.0/8 - This block is assigned for use as the Internet host loopback address
var ipv4LoopbackStart = net.IPv4(127, 0, 0, 0)

// NewNodeAddressHandler takes an IP family and the CIDR strings (
// NodePortAddresses, HealthzBindAddresses or MetricsBindAddresses, which is
// assumed to contain only valid CIDRs, potentially of both IP families) and returns a
// NodeAddressHandler object for the given family. If there are no CIDRs of the given
// family then the CIDR "0.0.0.0/0" or "::/0" will be added (even if there are CIDRs of
// the other family).
func NewNodeAddressHandler(family v1.IPFamily, cidrStrings []string) *NodeAddressHandler {
	nah := &NodeAddressHandler{}

	// Filter CIDRs to correct family
	for _, str := range cidrStrings {
		if (family == v1.IPv4Protocol) == netutils.IsIPv4CIDRString(str) {
			nah.cidrStrings = append(nah.cidrStrings, str)
		}
	}
	if len(nah.cidrStrings) == 0 {
		if family == v1.IPv4Protocol {
			nah.cidrStrings = []string{IPv4ZeroCIDR}
		} else {
			nah.cidrStrings = []string{IPv6ZeroCIDR}
		}
	}

	// Now parse
	for _, str := range nah.cidrStrings {
		_, cidr, _ := netutils.ParseCIDRSloppy(str)

		if netutils.IsIPv4CIDR(cidr) {
			if cidr.IP.IsLoopback() || cidr.Contains(ipv4LoopbackStart) {
				nah.containsIPv4Loopback = true
			}
		}

		if IsZeroCIDR(str) {
			// Ignore everything else
			nah.cidrs = []*net.IPNet{cidr}
			nah.matchAll = true
			break
		}

		nah.cidrs = append(nah.cidrs, cidr)
	}

	return nah
}

func (nah *NodeAddressHandler) String() string {
	return fmt.Sprintf("%v", nah.cidrStrings)
}

// MatchAll returns true if nah matches all node IPs (of nah's given family)
func (nah *NodeAddressHandler) MatchAll() bool {
	return nah.matchAll
}

// GetNodeIPs return all matched node IP addresses for nah's CIDRs. If no matching
// IPs are found, it returns an empty list.
// NetworkInterfacer is injected for test purpose.
func (nah *NodeAddressHandler) GetNodeIPs(nw NetworkInterfacer) ([]net.IP, error) {
	addrs, err := nw.InterfaceAddrs()
	if err != nil {
		return nil, fmt.Errorf("error listing all interfaceAddrs from host, error: %v", err)
	}

	// Use a map to dedup matches
	addresses := make(map[string]net.IP)
	for _, cidr := range nah.cidrs {
		for _, addr := range addrs {
			var ip net.IP
			// nw.InterfaceAddrs may return net.IPAddr or net.IPNet on windows, and it will return net.IPNet on linux.
			switch v := addr.(type) {
			case *net.IPAddr:
				ip = v.IP
			case *net.IPNet:
				ip = v.IP
			default:
				continue
			}

			if cidr.Contains(ip) {
				addresses[ip.String()] = ip
			}
		}
	}

	ips := make([]net.IP, 0, len(addresses))
	for _, ip := range addresses {
		ips = append(ips, ip)
	}

	return ips, nil
}

// ContainsIPv4Loopback returns true if nah's CIDRs contain an IPv4 loopback address.
func (nah *NodeAddressHandler) ContainsIPv4Loopback() bool {
	return nah.containsIPv4Loopback
}
