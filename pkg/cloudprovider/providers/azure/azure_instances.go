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

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
)

// NodeAddresses returns the addresses of the specified instance.
func (az *Cloud) NodeAddresses(name types.NodeName) ([]v1.NodeAddress, error) {
	ip, err := az.GetIPForMachineWithRetry(name)
	if err != nil {
		glog.V(2).Infof("NodeAddresses(%s) abort backoff", name)
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
	name, err := az.vmSet.GetNodeNameByProviderID(providerID)
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
	name, err := az.vmSet.GetNodeNameByProviderID(providerID)
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

// InstanceID returns the cloud provider ID of the specified instance.
// Note that if the instance does not exist or is no longer running, we must return ("", cloudprovider.InstanceNotFound)
func (az *Cloud) InstanceID(name types.NodeName) (string, error) {
	return az.vmSet.GetInstanceIDByNodeName(string(name))
}

// InstanceTypeByProviderID returns the cloudprovider instance type of the node with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (az *Cloud) InstanceTypeByProviderID(providerID string) (string, error) {
	name, err := az.vmSet.GetNodeNameByProviderID(providerID)
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
	return az.vmSet.GetInstanceTypeByNodeName(string(name))
}

// AddSSHKeyToAllInstances adds an SSH public key as a legal identity for all instances
// expected format for the key is standard ssh-keygen format: <protocol> <blob>
func (az *Cloud) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return fmt.Errorf("not supported")
}

// CurrentNodeName returns the name of the node we are currently running on.
// On Azure this is the hostname, so we just return the hostname.
func (az *Cloud) CurrentNodeName(hostname string) (types.NodeName, error) {
	return types.NodeName(hostname), nil
}

// mapNodeNameToVMName maps a k8s NodeName to an Azure VM Name
// This is a simple string cast.
func mapNodeNameToVMName(nodeName types.NodeName) string {
	return string(nodeName)
}
