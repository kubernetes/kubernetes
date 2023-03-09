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

// NewNodePortAddresses takes the `--nodeport-addresses` value (which is assumed to
// contain only valid CIDRs) and returns a NodePortAddresses object. If cidrStrings is
// empty, this is treated as `["0.0.0.0/0", "::/0"]`.
func NewNodePortAddresses(cidrStrings []string) *NodePortAddresses {
	if len(cidrStrings) == 0 {
		cidrStrings = []string{IPv4ZeroCIDR, IPv6ZeroCIDR}
	}

	npa := &NodePortAddresses{
		cidrStrings: cidrStrings,
	}

	for _, str := range npa.cidrStrings {
		_, cidr, _ := netutils.ParseCIDRSloppy(str)
		npa.cidrs = append(npa.cidrs, cidr)

		if netutils.IsIPv4CIDR(cidr) {
			if cidr.IP.IsLoopback() || cidr.Contains(ipv4LoopbackStart) {
				npa.containsIPv4Loopback = true
			}
		}
	}

	return npa
}

func (npa *NodePortAddresses) String() string {
	return fmt.Sprintf("%v", npa.cidrStrings)
}

// GetNodeAddresses return all matched node IP addresses for npa's CIDRs.
// If npa's CIDRs include "0.0.0.0/0" and/or "::/0", then those values will be returned
// verbatim in the response and no actual IPs of that family will be returned.
// If no matching IPs are found, GetNodeAddresses will return an error.
// NetworkInterfacer is injected for test purpose.
func (npa *NodePortAddresses) GetNodeAddresses(nw NetworkInterfacer) (sets.String, error) {
	uniqueAddressList := sets.NewString()

	// First round of iteration to pick out `0.0.0.0/0` or `::/0` for the sake of excluding non-zero IPs.
	for _, cidr := range npa.cidrStrings {
		if IsZeroCIDR(cidr) {
			uniqueAddressList.Insert(cidr)
		}
	}

	addrs, err := nw.InterfaceAddrs()
	if err != nil {
		return nil, fmt.Errorf("error listing all interfaceAddrs from host, error: %v", err)
	}

	// Second round of iteration to parse IPs based on cidr.
	for _, cidr := range npa.cidrs {
		if IsZeroCIDR(cidr.String()) {
			continue
		}

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
				if netutils.IsIPv6(ip) && !uniqueAddressList.Has(IPv6ZeroCIDR) {
					uniqueAddressList.Insert(ip.String())
				}
				if !netutils.IsIPv6(ip) && !uniqueAddressList.Has(IPv4ZeroCIDR) {
					uniqueAddressList.Insert(ip.String())
				}
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
