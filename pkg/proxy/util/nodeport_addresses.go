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

// NodePortAddresses is used to handle the --nodeport-addresses and
// --iptables-localhost-nodeports flags
type NodePortAddresses struct {
	cidrStrings []string

	cidrs          []*net.IPNet
	allowLocalhost bool
	matchAll       bool
}

// RFC 5735 127.0.0.0/8 - This block is assigned for use as the Internet host loopback address
var ipv4LoopbackStart = net.IPv4(127, 0, 0, 0)

// NewNodePortAddresses takes an IP family and the `--nodeport-addresses` value (which is
// assumed to contain only valid CIDRs, potentially of both IP families) and returns a
// NodePortAddresses object for the given family. If there are no CIDRs of the given
// family then the CIDR "0.0.0.0/0" or "::/0" will be added (even if there are CIDRs of
// the other family). Loopback IPs will be disallowed unless family is v1.IPv4Protocol and
// allowIPv4Localhost is true.
func NewNodePortAddresses(family v1.IPFamily, cidrStrings []string, allowIPv4Localhost bool) *NodePortAddresses {
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

		if family == v1.IPv4Protocol && allowIPv4Localhost && (cidr.IP.IsLoopback() || cidr.Contains(ipv4LoopbackStart)) {
			npa.allowLocalhost = true
		}

		if IsZeroCIDR(str) {
			// Ignore everything else
			npa.cidrs = []*net.IPNet{cidr}
			npa.matchAll = true
			break
		}

		npa.cidrs = append(npa.cidrs, cidr)
	}

	return npa
}

func (npa *NodePortAddresses) String() string {
	return fmt.Sprintf("%v", npa.cidrStrings)
}

// MatchAll returns true if npa matches all node IPs (of npa's given family)
func (npa *NodePortAddresses) MatchAll() bool {
	return npa.matchAll
}

// GetNodeAddresses return all matched node IP addresses for npa's CIDRs. If no matching
// IPs are found, it returns an empty list.
// NetworkInterfacer is injected for test purpose.
func (npa *NodePortAddresses) GetNodeAddresses(nw NetworkInterfacer) ([]string, error) {
	addrs, err := nw.InterfaceAddrs()
	if err != nil {
		return nil, fmt.Errorf("error listing all interfaceAddrs from host, error: %v", err)
	}

	uniqueAddressList := sets.NewString()
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

			if ip.IsLoopback() && !npa.allowLocalhost {
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

	return uniqueAddressList.List(), nil
}

// AllowLocalhost returns true if npa allows localhost NodePort connections in its IP
// family. Localhost NodePort connections are never allowed for IPv6. For IPv4, they
// are allowed only if npa was constructed with allowIPv4Localhost=true and a set of
// CIDRs that includes loopback IPs.
func (npa *NodePortAddresses) AllowLocalhost() bool {
	return npa.allowLocalhost
}
