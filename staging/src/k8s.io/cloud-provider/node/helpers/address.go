/*
Copyright 2019 The Kubernetes Authors.

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

package helpers

import (
	"fmt"
	"net"

	"k8s.io/api/core/v1"
	nodeutil "k8s.io/component-helpers/node/util"
	netutils "k8s.io/utils/net"
)

// AddToNodeAddresses appends the NodeAddresses to the passed-by-pointer slice,
// only if they do not already exist
func AddToNodeAddresses(addresses *[]v1.NodeAddress, addAddresses ...v1.NodeAddress) {
	for _, add := range addAddresses {
		exists := false
		for _, existing := range *addresses {
			if existing.Address == add.Address && existing.Type == add.Type {
				exists = true
				break
			}
		}
		if !exists {
			*addresses = append(*addresses, add)
		}
	}
}

// GetNodeAddressesFromNodeIPLegacy filters node addresses to prefer a specific node IP or
// address family. This function is used only with legacy cloud providers.
//
// If nodeIP is either '0.0.0.0' or '::' it is taken to represent any address of
// that address family: IPv4 or IPv6. i.e. if nodeIP is '0.0.0.0' we will return
// node addresses sorted such that all IPv4 addresses are listed before IPv6
// addresses.
//
// If nodeIP is a specific IP, either IPv4 or IPv6, we will return node
// addresses filtered such that:
//   - Any address matching nodeIP will be listed first.
//   - If nodeIP matches an address of a particular type (internal or external),
//     that will be the *only* address of that type returned.
//   - All remaining addresses are listed after.
func GetNodeAddressesFromNodeIPLegacy(nodeIP net.IP, cloudNodeAddresses []v1.NodeAddress) ([]v1.NodeAddress, error) {
	// If nodeIP is unset, just use the addresses provided by the cloud provider as-is
	if nodeIP == nil {
		return cloudNodeAddresses, nil
	}

	// nodeIP is "0.0.0.0" or "::"; sort cloudNodeAddresses to
	// prefer addresses of the matching family
	if nodeIP.IsUnspecified() {
		preferIPv4 := nodeIP.To4() != nil
		isPreferredIPFamily := func(ip net.IP) bool { return (ip.To4() != nil) == preferIPv4 }

		sortedAddresses := make([]v1.NodeAddress, 0, len(cloudNodeAddresses))
		for _, nodeAddress := range cloudNodeAddresses {
			ip := netutils.ParseIPSloppy(nodeAddress.Address)
			if ip == nil || isPreferredIPFamily(ip) {
				sortedAddresses = append(sortedAddresses, nodeAddress)
			}
		}
		for _, nodeAddress := range cloudNodeAddresses {
			ip := netutils.ParseIPSloppy(nodeAddress.Address)
			if ip != nil && !isPreferredIPFamily(ip) {
				sortedAddresses = append(sortedAddresses, nodeAddress)
			}
		}
		return sortedAddresses, nil
	}

	// Otherwise the result is the same as for GetNodeAddressesFromNodeIP
	return GetNodeAddressesFromNodeIP(nodeIP.String(), cloudNodeAddresses)
}

// GetNodeAddressesFromNodeIP filters the provided list of nodeAddresses to match the
// providedNodeIP from the Node annotation (which is assumed to be non-empty). This is
// used for external cloud providers.
//
// It will return node addresses filtered such that:
//   - Any address matching nodeIP will be listed first.
//   - If nodeIP matches an address of a particular type (internal or external),
//     that will be the *only* address of that type returned.
//   - All remaining addresses are listed after.
//
// (This does not have the same behavior with `0.0.0.0` and `::` as
// GetNodeAddressesFromNodeIPLegacy, because that case never occurs for external cloud
// providers, because kubelet does not set the `provided-node-ip` annotation in that
// case.)
func GetNodeAddressesFromNodeIP(providedNodeIP string, cloudNodeAddresses []v1.NodeAddress) ([]v1.NodeAddress, error) {
	nodeIPs, err := nodeutil.ParseNodeIPAnnotation(providedNodeIP)
	if err != nil {
		return nil, fmt.Errorf("failed to parse node IP %q: %v", providedNodeIP, err)
	}

	enforcedNodeAddresses := []v1.NodeAddress{}
	nodeIPTypes := make(map[v1.NodeAddressType]bool)

	for _, nodeIP := range nodeIPs {
		// For every address supplied by the cloud provider that matches nodeIP,
		// nodeIP is the enforced node address for that address Type (like
		// InternalIP and ExternalIP), meaning other addresses of the same Type
		// are discarded. See #61921 for more information: some cloud providers
		// may supply secondary IPs, so nodeIP serves as a way to ensure that the
		// correct IPs show up on a Node object.

		matched := false
		for _, nodeAddress := range cloudNodeAddresses {
			if netutils.ParseIPSloppy(nodeAddress.Address).Equal(nodeIP) {
				enforcedNodeAddresses = append(enforcedNodeAddresses, v1.NodeAddress{Type: nodeAddress.Type, Address: nodeAddress.Address})
				nodeIPTypes[nodeAddress.Type] = true
				matched = true
			}
		}

		// nodeIP must be among the addresses supplied by the cloud provider
		if !matched {
			return nil, fmt.Errorf("failed to get node address from cloud provider that matches ip: %v", nodeIP)
		}
	}

	// Now use all other addresses supplied by the cloud provider NOT of the same Type
	// as any nodeIP.
	for _, nodeAddress := range cloudNodeAddresses {
		if !nodeIPTypes[nodeAddress.Type] {
			enforcedNodeAddresses = append(enforcedNodeAddresses, v1.NodeAddress{Type: nodeAddress.Type, Address: nodeAddress.Address})
		}
	}

	return enforcedNodeAddresses, nil
}
