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

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
)

// NodeAddresses returns the addresses of the specified instance.
func (az *Cloud) NodeAddresses(name types.NodeName) ([]v1.NodeAddress, error) {
	if az.UseInstanceMetadata {
		ipAddress := IPAddress{}
		err := az.metadata.Object("instance/network/interface/0/ipv4/ipAddress/0", &ipAddress)
		if err != nil {
			return nil, err
		}
		addresses := []v1.NodeAddress{
			{Type: v1.NodeInternalIP, Address: ipAddress.PrivateIP},
			{Type: v1.NodeHostName, Address: string(name)},
		}
		if len(ipAddress.PublicIP) > 0 {
			addr := v1.NodeAddress{
				Type:    v1.NodeExternalIP,
				Address: ipAddress.PublicIP,
			}
			addresses = append(addresses, addr)
		}
		return addresses, nil
	}
	ip, err := az.getIPForMachine(name)
	if err != nil {
		if az.CloudProviderBackoff {
			glog.V(2).Infof("NodeAddresses(%s) backing off", name)
			ip, err = az.GetIPForMachineWithRetry(name)
			if err != nil {
				glog.V(2).Infof("NodeAddresses(%s) abort backoff", name)
				return nil, err
			}
		} else {
			glog.Errorf("error: az.NodeAddresses, az.getIPForMachine(%s), err=%v", name, err)
			return nil, err
		}
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
	name, err := splitProviderID(providerID)
	if err != nil {
		return nil, err
	}

	return az.NodeAddresses(name)
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (az *Cloud) ExternalID(name types.NodeName) (string, error) {
	return az.InstanceID(name)
}

// InstanceExistsByProviderID returns true if the instance with the given provider id still exists and is running.
// If false is returned with no error, the instance will be immediately deleted by the cloud controller manager.
func (az *Cloud) InstanceExistsByProviderID(providerID string) (bool, error) {
	return false, errors.New("unimplemented")
}

func (az *Cloud) isCurrentInstance(name types.NodeName) (bool, error) {
	nodeName := mapNodeNameToVMName(name)
	metadataName, err := az.metadata.Text("instance/compute/name")
	return (metadataName == nodeName), err
}

// InstanceID returns the cloud provider ID of the specified instance.
// Note that if the instance does not exist or is no longer running, we must return ("", cloudprovider.InstanceNotFound)
func (az *Cloud) InstanceID(name types.NodeName) (string, error) {
	if az.UseInstanceMetadata {
		isLocalInstance, err := az.isCurrentInstance(name)
		if err != nil {
			return "", err
		}
		if isLocalInstance {
			externalInstanceID, err := az.metadata.Text("instance/compute/vmId")
			if err == nil {
				return externalInstanceID, nil
			}
		}
	}
	var machine compute.VirtualMachine
	var exists bool
	var err error
	az.operationPollRateLimiter.Accept()
	machine, exists, err = az.getVirtualMachine(name)
	if err != nil {
		if az.CloudProviderBackoff {
			glog.V(2).Infof("InstanceID(%s) backing off", name)
			machine, exists, err = az.GetVirtualMachineWithRetry(name)
			if err != nil {
				glog.V(2).Infof("InstanceID(%s) abort backoff", name)
				return "", err
			}
		} else {
			return "", err
		}
	} else if !exists {
		return "", cloudprovider.InstanceNotFound
	}
	return *machine.ID, nil
}

// InstanceTypeByProviderID returns the cloudprovider instance type of the node with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (az *Cloud) InstanceTypeByProviderID(providerID string) (string, error) {
	name, err := splitProviderID(providerID)
	if err != nil {
		return "", err
	}

	return az.InstanceType(name)
}

// InstanceType returns the type of the specified instance.
// Note that if the instance does not exist or is no longer running, we must return ("", cloudprovider.InstanceNotFound)
// (Implementer Note): This is used by kubelet. Kubelet will label the node. Real log from kubelet:
//       Adding node label from cloud provider: beta.kubernetes.io/instance-type=[value]
func (az *Cloud) InstanceType(name types.NodeName) (string, error) {
	if az.UseInstanceMetadata {
		isLocalInstance, err := az.isCurrentInstance(name)
		if err != nil {
			return "", err
		}
		if isLocalInstance {
			machineType, err := az.metadata.Text("instance/compute/vmSize")
			if err == nil {
				return machineType, nil
			}
		}
	}
	machine, exists, err := az.getVirtualMachine(name)
	if err != nil {
		glog.Errorf("error: az.InstanceType(%s), az.getVirtualMachine(%s) err=%v", name, name, err)
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

	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("VirtualMachinesClient.List(%s): start", az.ResourceGroup)
	result, err := az.VirtualMachinesClient.List(az.ResourceGroup)
	glog.V(10).Infof("VirtualMachinesClient.List(%s): end", az.ResourceGroup)
	if err != nil {
		glog.Errorf("error: az.listAllNodesInResourceGroup(), az.VirtualMachinesClient.List(%s), err=%v", az.ResourceGroup, err)
		return nil, err
	}

	morePages := (result.Value != nil && len(*result.Value) > 1)

	for morePages {
		allNodes = append(allNodes, *result.Value...)

		az.operationPollRateLimiter.Accept()
		glog.V(10).Infof("VirtualMachinesClient.ListAllNextResults(%v): start", az.ResourceGroup)
		result, err = az.VirtualMachinesClient.ListAllNextResults(result)
		glog.V(10).Infof("VirtualMachinesClient.ListAllNextResults(%v): end", az.ResourceGroup)
		if err != nil {
			glog.Errorf("error: az.listAllNodesInResourceGroup(), az.VirtualMachinesClient.ListAllNextResults(%v), err=%v", result, err)
			return nil, err
		}

		morePages = (result.Value != nil && len(*result.Value) > 1)
	}

	return allNodes, nil

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
