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
	"errors"
	"fmt"
	"regexp"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"k8s.io/apimachinery/pkg/types"
)

// NodeAddresses returns the addresses of the specified instance.
func (az *Cloud) NodeAddresses(name types.NodeName) ([]v1.NodeAddress, error) {
	ip, err := az.getIPForMachine(name)
	if err != nil {
		return nil, err
	}

	return []v1.NodeAddress{
		{Type: v1.NodeInternalIP, Address: ip},
		{Type: v1.NodeHostName, Address: string(name)},
	}, nil
}

// NodeAddressesByProviderID returns the node addresses of an instances with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (az *Cloud) NodeAddressesByProviderID(providerID string) ([]v1.NodeAddress, error) {
	return []v1.NodeAddress{}, errors.New("unimplemented")
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

// InstanceTypeByProviderID returns the cloudprovider instance type of the node with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (az *Cloud) InstanceTypeByProviderID(providerID string) (string, error) {
	return "", errors.New("unimplemented")
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
