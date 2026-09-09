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

package util

import (
	"fmt"
	"net"
	"strings"

	netutils "k8s.io/utils/net"
)

const (
	cloudProviderNone     = ""
	cloudProviderExternal = "external"
)

// parseNodeIP implements ParseNodeIPArgument and ParseNodeIPAnnotation
func parseNodeIP(nodeIP string, allowDual, sloppy bool) ([]net.IP, []string, error) {
	var nodeIPs []net.IP
	var invalidIPs []string
	if nodeIP != "" || !sloppy {
		for _, ip := range strings.Split(nodeIP, ",") {
			if sloppy {
				ip = strings.TrimSpace(ip)
			}
			parsedNodeIP := netutils.ParseIPSloppy(ip)
			if parsedNodeIP == nil {
				invalidIPs = append(invalidIPs, ip)
				if !sloppy {
					return nil, invalidIPs, fmt.Errorf("could not parse %q", ip)
				}
			} else {
				nodeIPs = append(nodeIPs, parsedNodeIP)
			}
		}
	}

	if len(nodeIPs) > 2 || (len(nodeIPs) == 2 && netutils.IsIPv6(nodeIPs[0]) == netutils.IsIPv6(nodeIPs[1])) {
		return nil, invalidIPs, fmt.Errorf("must contain either a single IP or a dual-stack pair of IPs")
	} else if len(nodeIPs) == 2 && !allowDual {
		return nil, invalidIPs, fmt.Errorf("dual-stack not supported in this configuration")
	} else if len(nodeIPs) == 2 && (nodeIPs[0].IsUnspecified() || nodeIPs[1].IsUnspecified()) {
		return nil, invalidIPs, fmt.Errorf("dual-stack node IP cannot include '0.0.0.0' or '::'")
	}

	return nodeIPs, invalidIPs, nil
}

// ParseNodeIPArgument parses kubelet's --node-ip argument.
// If nodeIP contains invalid values, they will be returned as strings.
// This is done also when an error is returned.
// The caller then can decide what to do with the invalid values.
// Dual-stack node IPs are allowed if cloudProvider is unset or `"external"`.
func ParseNodeIPArgument(nodeIP, cloudProvider string) ([]net.IP, []string, error) {
	var allowDualStack bool
	if cloudProvider == cloudProviderNone || cloudProvider == cloudProviderExternal {
		allowDualStack = true
	}
	return parseNodeIP(nodeIP, allowDualStack, true)
}

// ParseNodeIPAnnotation parses the `alpha.kubernetes.io/provided-node-ip` annotation,
// which can be either a single IP address or a comma-separated pair of IP addresses.
// Unlike with ParseNodeIPArgument, invalid values are considered an error.
func ParseNodeIPAnnotation(nodeIP string) ([]net.IP, error) {
	nodeIps, _, err := parseNodeIP(nodeIP, true, false)
	return nodeIps, err
}
