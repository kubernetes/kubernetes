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
	"k8s.io/apimachinery/pkg/util/sets"
	netutils "k8s.io/utils/net"
)

// NodePortAddresses is used to handle the --nodeport-addresses flag
type NodePortAddresses struct {
	cidrStrings []string

	cidrs                []*net.IPNet
	containsIPv4Loopback bool
}

// RFC 5735 127.0.0.0/8 - This block is assigned for use as the Internet host loopback address
var ipv4LoopbackStart = net.IPv4(127, 0, 0, 0)

// NewNodePortAddresses takes an IP family and the `--nodeport-addresses` value (which is
// assumed to contain only valid CIDRs, potentially of both IP families) and returns a
// NodePortAddresses object for the given family. If there are no CIDRs of the given
// family then the CIDR "0.0.0.0/0" or "::/0" will be added (even if there are CIDRs of
// the other family).
func NewNodePortAddresses(family v1.IPFamily, cidrStrings []string) *NodePortAddresses {
	npa := &NodePortAddresses{}

	// Filter CIDRs to correct family
	for _, str := range cidrStrings {
		if (family == v1.IPv4Protocol) == netutils.IsIPv4CIDRString(str) {
			npa.cidrStrings = append(npa.cidrStrings, str)
		}
	}
	if len(npa.cidrStrings) == 0 {
		if family == v1.IPv4Protocol {
			npa.cidrStrings = []string{IPv4ZeroCIDR}
		} else {
			npa.cidrStrings = []string{IPv6ZeroCIDR}
		}
	}

	// Now parse
	for _, str := range npa.cidrStrings {
		_, cidr, _ := netutils.ParseCIDRSloppy(str)

		if netutils.IsIPv4CIDR(cidr) {
			if cidr.IP.IsLoopback() || cidr.Contains(ipv4LoopbackStart) {
				npa.containsIPv4Loopback = true
			}
		}

		if IsZeroCIDR(str) {
			// Ignore everything else
			npa.cidrs = []*net.IPNet{cidr}
			break
		}

		npa.cidrs = append(npa.cidrs, cidr)
	}

	return npa
}

func (npa *NodePortAddresses) String() string {
	return fmt.Sprintf("%v", npa.cidrStrings)
}

// GetNodeAddresses returns all matched node IP addresses for npa's IP family. If npa's
// CIDRs include "0.0.0.0/0" or "::/0", then that value will be returned verbatim in
// the response and no actual IPs of that family will be returned. If no matching IPs are
// found, GetNodeAddresses will return an error.
// NetworkInterfacer is injected for test purpose.
func (npa *NodePortAddresses) GetNodeAddresses(nw NetworkInterfacer) (sets.Set[string], error) {
	uniqueAddressList := sets.New[string]()

	// First round of iteration to pick out `0.0.0.0/0` or `::/0` for the sake of excluding non-zero IPs.
	for _, cidr := range npa.cidrStrings {
		if IsZeroCIDR(cidr) {
			uniqueAddressList.Insert(cidr)
			return uniqueAddressList, nil
		}
	}

	addrs, err := nw.InterfaceAddrs()
	if err != nil {
		return nil, fmt.Errorf("error listing all interfaceAddrs from host, error: %v", err)
	}

	for _, cidr := range npa.cidrs {
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
				uniqueAddressList.Insert(ip.String())
			}
		}
	}

	if uniqueAddressList.Len() == 0 {
		return nil, fmt.Errorf("no addresses found for cidrs %v", npa.cidrStrings)
	}

	return uniqueAddressList, nil
}

// ContainsIPv4Loopback returns true if npa's CIDRs contain an IPv4 loopback address.
func (npa *NodePortAddresses) ContainsIPv4Loopback() bool {
	return npa.containsIPv4Loopback
}
