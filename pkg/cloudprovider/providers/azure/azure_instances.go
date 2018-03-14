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

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
)

// NodeAddresses returns the addresses of the specified instance.
func (az *Cloud) NodeAddresses(name types.NodeName) ([]v1.NodeAddress, error) {
	addressGetter := func(nodeName types.NodeName) ([]v1.NodeAddress, error) {
		ip, publicIP, err := az.GetIPForMachineWithRetry(nodeName)
		if err != nil {
			glog.V(2).Infof("NodeAddresses(%s) abort backoff", nodeName)
			return nil, err
		}

		addresses := []v1.NodeAddress{
			{Type: v1.NodeInternalIP, Address: ip},
			{Type: v1.NodeHostName, Address: string(name)},
		}
		if len(publicIP) > 0 {
			addresses = append(addresses, v1.NodeAddress{
				Type:    v1.NodeExternalIP,
				Address: publicIP,
			})
		}
		return addresses, nil
	}

	if az.UseInstanceMetadata {
		isLocalInstance, err := az.isCurrentInstance(name)
		if err != nil {
			return nil, err
		}

		// Not local instance, get addresses from Azure ARM API.
		if !isLocalInstance {
			return addressGetter(name)
		}

		ipAddress := IPAddress{}
		err = az.metadata.Object("instance/network/interface/0/ipv4/ipAddress/0", &ipAddress)
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

	return addressGetter(name)
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
	name, err := splitProviderID(providerID)
	if err != nil {
		return false, err
	}

	_, err = az.InstanceID(name)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			return false, nil
		}
		return false, err
	}

	return true, nil
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
			nodeName := mapNodeNameToVMName(name)
			return az.getMachineID(nodeName), nil
		}
	}

	if az.Config.VMType == vmTypeVMSS {
		id, err := az.getVmssInstanceID(name)
		if err == cloudprovider.InstanceNotFound || err == ErrorNotVmssInstance {
			// Retry with standard type because master nodes may not belong to any vmss.
			return az.getStandardInstanceID(name)
		}

		return id, err
	}

	return az.getStandardInstanceID(name)
}

func (az *Cloud) getVmssInstanceID(name types.NodeName) (string, error) {
	var machine compute.VirtualMachineScaleSetVM
	var exists bool
	var err error
	az.operationPollRateLimiter.Accept()
	machine, exists, err = az.getVmssVirtualMachine(name)
	if err != nil {
		if az.CloudProviderBackoff {
			glog.V(2).Infof("InstanceID(%s) backing off", name)
			machine, exists, err = az.GetScaleSetsVMWithRetry(name)
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

func (az *Cloud) getStandardInstanceID(name types.NodeName) (string, error) {
	var machine compute.VirtualMachine
	var err error
	az.operationPollRateLimiter.Accept()
	machine, err = az.getVirtualMachine(name)
	if err != nil {
		if az.CloudProviderBackoff {
			glog.V(2).Infof("InstanceID(%s) backing off", name)
			machine, err = az.GetVirtualMachineWithRetry(name)
			if err != nil {
				glog.V(2).Infof("InstanceID(%s) abort backoff", name)
				return "", err
			}
		} else {
			return "", err
		}
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

	if az.Config.VMType == vmTypeVMSS {
		machineType, err := az.getVmssInstanceType(name)
		if err == cloudprovider.InstanceNotFound || err == ErrorNotVmssInstance {
			// Retry with standard type because master nodes may not belong to any vmss.
			return az.getStandardInstanceType(name)
		}

		return machineType, err
	}

	return az.getStandardInstanceType(name)
}

// getVmssInstanceType gets instance with type vmss.
func (az *Cloud) getVmssInstanceType(name types.NodeName) (string, error) {
	machine, exists, err := az.getVmssVirtualMachine(name)
	if err != nil {
		glog.Errorf("error: az.InstanceType(%s), az.getVmssVirtualMachine(%s) err=%v", name, name, err)
		return "", err
	} else if !exists {
		return "", cloudprovider.InstanceNotFound
	}

	if machine.Sku.Name != nil {
		return *machine.Sku.Name, nil
	}

	return "", fmt.Errorf("instance type is not set")
}

// getStandardInstanceType gets instance with standard type.
func (az *Cloud) getStandardInstanceType(name types.NodeName) (string, error) {
	machine, err := az.getVirtualMachine(name)
	if err != nil {
		glog.Errorf("error: az.InstanceType(%s), az.getVirtualMachine(%s) err=%v", name, name, err)
		return "", err
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
