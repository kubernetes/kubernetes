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

// GetNodeAddresses return all matched node IP addresses based on given cidr slice.
// Some callers, e.g. IPVS proxier, need concrete IPs, not ranges, which is why this exists.
// NetworkInterfacer is injected for test purpose.
// We expect the cidrs passed in is already validated.
// Given an empty input `[]`, it will return `0.0.0.0/0` and `::/0` directly.
// If multiple cidrs is given, it will return the minimal IP sets, e.g. given input `[1.2.0.0/16, 0.0.0.0/0]`, it will
// only return `0.0.0.0/0`.
// NOTE: GetNodeAddresses only accepts CIDRs, if you want concrete IPs, e.g. 1.2.3.4, then the input should be 1.2.3.4/32.
func GetNodeAddresses(cidrs []string, nw NetworkInterfacer) (sets.String, error) {
	uniqueAddressList := sets.NewString()
	if len(cidrs) == 0 {
		uniqueAddressList.Insert(IPv4ZeroCIDR)
		uniqueAddressList.Insert(IPv6ZeroCIDR)
		return uniqueAddressList, nil
	}
	// First round of iteration to pick out `0.0.0.0/0` or `::/0` for the sake of excluding non-zero IPs.
	for _, cidr := range cidrs {
		if IsZeroCIDR(cidr) {
			uniqueAddressList.Insert(cidr)
		}
	}

	addrs, err := nw.InterfaceAddrs()
	if err != nil {
		return nil, fmt.Errorf("error listing all interfaceAddrs from host, error: %v", err)
	}

	// Second round of iteration to parse IPs based on cidr.
	for _, cidr := range cidrs {
		if IsZeroCIDR(cidr) {
			continue
		}

		_, ipNet, _ := netutils.ParseCIDRSloppy(cidr)
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

			if ipNet.Contains(ip) {
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
		return nil, fmt.Errorf("no addresses found for cidrs %v", cidrs)
	}

	return uniqueAddressList, nil
}

// ContainsIPv4Loopback returns true if the input is empty or one of the CIDR contains an IPv4 loopback address.
func ContainsIPv4Loopback(cidrStrings []string) bool {
	if len(cidrStrings) == 0 {
		return true
	}
	// RFC 5735 127.0.0.0/8 - This block is assigned for use as the Internet host loopback address
	ipv4LoopbackStart := netutils.ParseIPSloppy("127.0.0.0")
	for _, cidr := range cidrStrings {
		ip, ipnet, err := netutils.ParseCIDRSloppy(cidr)
		if err != nil {
			continue
		}

		if netutils.IsIPv6CIDR(ipnet) {
			continue
		}

		if ip.IsLoopback() {
			return true
		}
		if ipnet.Contains(ipv4LoopbackStart) {
			return true
		}
	}
	return false
}
