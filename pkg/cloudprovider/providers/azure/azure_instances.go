/*
Copyright 2016 The Kubernetes Authors.

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

package azure

import (
	"fmt"
	"regexp"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"k8s.io/kubernetes/pkg/types"
)

// NodeAddresses returns the addresses of the specified instance.
func (az *Cloud) NodeAddresses(name types.NodeName) ([]api.NodeAddress, error) {
	ip, err := az.getIPForMachine(name)
	if err != nil {
		return nil, err
	}

	return []api.NodeAddress{
		{Type: api.NodeInternalIP, Address: ip},
		{Type: api.NodeHostName, Address: string(name)},
	}, nil
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (az *Cloud) ExternalID(name types.NodeName) (string, error) {
	return az.InstanceID(name)
}

// InstanceID returns the cloud provider ID of the specified instance.
// Note that if the instance does not exist or is no longer running, we must return ("", cloudprovider.InstanceNotFound)
func (az *Cloud) InstanceID(name types.NodeName) (string, error) {
	machine, exists, err := az.getVirtualMachine(name)
	if err != nil {
		return "", err
	} else if !exists {
		return "", cloudprovider.InstanceNotFound
	}
	return *machine.ID, nil
}

// InstanceType returns the type of the specified instance.
// Note that if the instance does not exist or is no longer running, we must return ("", cloudprovider.InstanceNotFound)
// (Implementer Note): This is used by kubelet. Kubelet will label the node. Real log from kubelet:
//       Adding node label from cloud provider: beta.kubernetes.io/instance-type=[value]
func (az *Cloud) InstanceType(name types.NodeName) (string, error) {
	machine, exists, err := az.getVirtualMachine(name)
	if err != nil {
		return "", err
	} else if !exists {
		return "", cloudprovider.InstanceNotFound
	}
	return string(machine.HardwareProfile.VMSize), nil
}

// List lists instances that match 'filter' which is a regular expression which must match the entire instance name (fqdn)
func (az *Cloud) List(filter string) ([]types.NodeName, error) {
	allNodes, err := az.listAllNodesInResourceGroup()
	if err != nil {
		return nil, err
	}

	filteredNodes, err := filterNodes(allNodes, filter)
	if err != nil {
		return nil, err
	}

	nodeNames := make([]types.NodeName, len(filteredNodes))
	for i, v := range filteredNodes {
		nodeNames[i] = types.NodeName(*v.Name)
	}

	return nodeNames, nil
}

// AddSSHKeyToAllInstances adds an SSH public key as a legal identity for all instances
// expected format for the key is standard ssh-keygen format: <protocol> <blob>
func (az *Cloud) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return fmt.Errorf("not supported")
}

// CurrentNodeName returns the name of the node we are currently running on
// On most clouds (e.g. GCE) this is the hostname, so we provide the hostname
func (az *Cloud) CurrentNodeName(hostname string) (types.NodeName, error) {
	return types.NodeName(hostname), nil
}

func (az *Cloud) listAllNodesInResourceGroup() ([]compute.VirtualMachine, error) {
	allNodes := []compute.VirtualMachine{}

	result, err := az.VirtualMachinesClient.List(az.ResourceGroup)
	if err != nil {
		return nil, err
	}

	morePages := (result.Value != nil && len(*result.Value) > 1)

	for morePages {
		allNodes = append(allNodes, *result.Value...)

		result, err = az.VirtualMachinesClient.ListAllNextResults(result)
		if err != nil {
			return nil, err
		}

		morePages = (result.Value != nil && len(*result.Value) > 1)
	}

	return allNodes, nil

}

func filterNodes(nodes []compute.VirtualMachine, filter string) ([]compute.VirtualMachine, error) {
	filteredNodes := []compute.VirtualMachine{}

	re, err := regexp.Compile(filter)
	if err != nil {
		return nil, err
	}

	for _, node := range nodes {
		// search tags
		if re.MatchString(*node.Name) {
			filteredNodes = append(filteredNodes, node)
		}
	}

	return filteredNodes, nil
}

// mapNodeNameToVMName maps a k8s NodeName to an Azure VM Name
// This is a simple string cast.
func mapNodeNameToVMName(nodeName types.NodeName) string {
	return string(nodeName)
}

// mapVMNameToNodeName maps an Azure VM Name to a k8s NodeName
// This is a simple string cast.
func mapVMNameToNodeName(vmName string) types.NodeName {
	return types.NodeName(vmName)
}
