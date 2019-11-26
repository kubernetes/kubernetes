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

package nodeaddress

import (
	"fmt"
	"net"

	v1 "k8s.io/api/core/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/kubelet/nodestatus"
)

// Plugin is the node status plugin.
type Plugin struct {
	nodeIP                net.IP
	validateNodeIPFunc    func(net.IP) error
	hostname              string
	hostnameOverridden    bool
	externalCloudProvider bool
	cloud                 cloudprovider.Interface
	nodeAddressesFunc     func() ([]v1.NodeAddress, error)
}

var _ nodestatus.Plugin = &Plugin{}

// New returns a node address plugin.
func New(
	nodeIP net.IP, // typically Kubelet.nodeIP
	validateNodeIPFunc func(net.IP) error, // typically Kubelet.nodeIPValidator
	hostname string, // typically Kubelet.hostname
	hostnameOverridden bool, // was the hostname force set?
	externalCloudProvider bool, // typically Kubelet.externalCloudProvider
	cloud cloudprovider.Interface, // typically Kubelet.cloud
	nodeAddressesFunc func() ([]v1.NodeAddress, error), // typically Kubelet.cloudResourceSyncManager.NodeAddresses
) *Plugin {
	return &Plugin{
		nodeIP:                nodeIP,
		validateNodeIPFunc:    validateNodeIPFunc,
		hostname:              hostname,
		hostnameOverridden:    hostnameOverridden,
		externalCloudProvider: externalCloudProvider,
		cloud:                 cloud,
		nodeAddressesFunc:     nodeAddressesFunc,
	}
}

// Name is the name of the plugin.
func (p *Plugin) Name() string {
	return nodestatus.NodeAddressPluginName
}

// Update updates the status of the node.
func (p *Plugin) Update(node *v1.Node) error {
	hostname := p.hostname
	nodeIP := p.nodeIP

	if nodeIP != nil {
		if err := p.validateNodeIPFunc(nodeIP); err != nil {
			return fmt.Errorf("failed to validate nodeIP: %v", err)
		}
		klog.V(2).Infof("Using node IP: %q", nodeIP.String())
	}

	if p.externalCloudProvider {
		if nodeIP != nil {
			if node.ObjectMeta.Annotations == nil {
				node.ObjectMeta.Annotations = make(map[string]string)
			}
			node.ObjectMeta.Annotations[kubeletapis.AnnotationProvidedIPAddr] = nodeIP.String()
		}

		// If --cloud-provider=external and node address is already set,
		// then we return early because provider set addresses should take precedence.
		// Otherwise, we try to look up the node IP and let the cloud provider override it later
		// This should alleviate a lot of the bootstrapping issues with out-of-tree providers
		if len(node.Status.Addresses) > 0 {
			return nil
		}
	}
	if p.cloud != nil {
		cloudNodeAddresses, err := p.nodeAddressesFunc()
		if err != nil {
			return err
		}

		var nodeAddresses []v1.NodeAddress

		// For every address supplied by the cloud provider that matches nodeIP, nodeIP is the enforced node address for
		// that address Type (like InternalIP and ExternalIP), meaning other addresses of the same Type are discarded.
		// See #61921 for more information: some cloud providers may supply secondary IPs, so nodeIP serves as a way to
		// ensure that the correct IPs show up on a Node object.
		if nodeIP != nil {
			enforcedNodeAddresses := []v1.NodeAddress{}

			nodeIPTypes := make(map[v1.NodeAddressType]bool)
			for _, nodeAddress := range cloudNodeAddresses {
				if nodeAddress.Address == nodeIP.String() {
					enforcedNodeAddresses = append(enforcedNodeAddresses, v1.NodeAddress{Type: nodeAddress.Type, Address: nodeAddress.Address})
					nodeIPTypes[nodeAddress.Type] = true
				}
			}

			// nodeIP must be among the addresses supplied by the cloud provider
			if len(enforcedNodeAddresses) == 0 {
				return fmt.Errorf("failed to get node address from cloud provider that matches ip: %v", nodeIP)
			}

			// nodeIP was found, now use all other addresses supplied by the cloud provider NOT of the same Type as nodeIP.
			for _, nodeAddress := range cloudNodeAddresses {
				if !nodeIPTypes[nodeAddress.Type] {
					enforcedNodeAddresses = append(enforcedNodeAddresses, v1.NodeAddress{Type: nodeAddress.Type, Address: nodeAddress.Address})
				}
			}

			nodeAddresses = enforcedNodeAddresses
		} else {
			// If nodeIP is unset, just use the addresses provided by the cloud provider as-is
			nodeAddresses = cloudNodeAddresses
		}

		switch {
		case len(cloudNodeAddresses) == 0:
			// the cloud provider didn't specify any addresses
			nodeAddresses = append(nodeAddresses, v1.NodeAddress{Type: v1.NodeHostName, Address: hostname})

		case !hasAddressType(cloudNodeAddresses, v1.NodeHostName) && hasAddressValue(cloudNodeAddresses, hostname):
			// the cloud provider didn't specify an address of type Hostname,
			// but the auto-detected hostname matched an address reported by the cloud provider,
			// so we can add it and count on the value being verifiable via cloud provider metadata
			nodeAddresses = append(nodeAddresses, v1.NodeAddress{Type: v1.NodeHostName, Address: hostname})

		case p.hostnameOverridden:
			// the hostname was force-set via flag/config.
			// this means the hostname might not be able to be validated via cloud provider metadata,
			// but was a choice by the kubelet deployer we should honor
			var existingHostnameAddress *v1.NodeAddress
			for i := range nodeAddresses {
				if nodeAddresses[i].Type == v1.NodeHostName {
					existingHostnameAddress = &nodeAddresses[i]
					break
				}
			}

			if existingHostnameAddress == nil {
				// no existing Hostname address found, add it
				klog.Warningf("adding overridden hostname of %v to cloudprovider-reported addresses", hostname)
				nodeAddresses = append(nodeAddresses, v1.NodeAddress{Type: v1.NodeHostName, Address: hostname})
			} else if existingHostnameAddress.Address != hostname {
				// override the Hostname address reported by the cloud provider
				klog.Warningf("replacing cloudprovider-reported hostname of %v with overridden hostname of %v", existingHostnameAddress.Address, hostname)
				existingHostnameAddress.Address = hostname
			}
		}
		node.Status.Addresses = nodeAddresses
	} else {
		var ipAddr net.IP
		var err error

		// 1) Use nodeIP if set
		// 2) If the user has specified an IP to HostnameOverride, use it
		// 3) Lookup the IP from node name by DNS and use the first valid IPv4 address.
		//    If the node does not have a valid IPv4 address, use the first valid IPv6 address.
		// 4) Try to get the IP from the network interface used as default gateway
		if nodeIP != nil {
			ipAddr = nodeIP
		} else if addr := net.ParseIP(hostname); addr != nil {
			ipAddr = addr
		} else {
			var addrs []net.IP
			addrs, _ = net.LookupIP(node.Name)
			for _, addr := range addrs {
				if err = p.validateNodeIPFunc(addr); err == nil {
					if addr.To4() != nil {
						ipAddr = addr
						break
					}
					if addr.To16() != nil && ipAddr == nil {
						ipAddr = addr
					}
				}
			}

			if ipAddr == nil {
				ipAddr, err = utilnet.ChooseHostInterface()
			}
		}

		if ipAddr == nil {
			// We tried everything we could, but the IP address wasn't fetchable; error out
			return fmt.Errorf("can't get ip address of node %s. error: %v", node.Name, err)
		}
		node.Status.Addresses = []v1.NodeAddress{
			{Type: v1.NodeInternalIP, Address: ipAddr.String()},
			{Type: v1.NodeHostName, Address: hostname},
		}
	}
	return nil
}

func hasAddressType(addresses []v1.NodeAddress, addressType v1.NodeAddressType) bool {
	for _, address := range addresses {
		if address.Type == addressType {
			return true
		}
	}
	return false
}

func hasAddressValue(addresses []v1.NodeAddress, addressValue string) bool {
	for _, address := range addresses {
		if address.Address == addressValue {
			return true
		}
	}
	return false
}
