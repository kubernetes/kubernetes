/*
Copyright 2017 The Kubernetes Authors.

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
	"strconv"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

func (az *Cloud) getIPForVmssMachine(nodeName types.NodeName) (string, error) {
	az.operationPollRateLimiter.Accept()
	machine, exists, err := az.getVmssVirtualMachine(nodeName)
	if !exists {
		return "", cloudprovider.InstanceNotFound
	}
	if err != nil {
		glog.Errorf("error: az.getIPForVmssMachine(%s), az.getVmssVirtualMachine(%s), err=%v", nodeName, nodeName, err)
		return "", err
	}

	nicID, err := getPrimaryInterfaceIDForVmssMachine(machine)
	if err != nil {
		glog.Errorf("error: az.getIPForVmssMachine(%s), getPrimaryInterfaceID(%v), err=%v", nodeName, machine, err)
		return "", err
	}

	nicName, err := getLastSegment(nicID)
	if err != nil {
		glog.Errorf("error: az.getIPForVmssMachine(%s), getLastSegment(%s), err=%v", nodeName, nicID, err)
		return "", err
	}

	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("InterfacesClient.Get(%q): start", nicName)
	nic, err := az.InterfacesClient.GetVirtualMachineScaleSetNetworkInterface(az.ResourceGroup, az.Config.PrimaryScaleSetName, *machine.InstanceID, nicName, "")
	glog.V(10).Infof("InterfacesClient.Get(%q): end", nicName)
	if err != nil {
		glog.Errorf("error: az.getIPForVmssMachine(%s), az.GetVirtualMachineScaleSetNetworkInterface.Get(%s, %s, %s), err=%v", nodeName, az.ResourceGroup, nicName, "", err)
		return "", err
	}

	ipConfig, err := getPrimaryIPConfig(nic)
	if err != nil {
		glog.Errorf("error: az.getIPForVmssMachine(%s), getPrimaryIPConfig(%v), err=%v", nodeName, nic, err)
		return "", err
	}

	targetIP := *ipConfig.PrivateIPAddress
	return targetIP, nil
}

// This returns the full identifier of the primary NIC for the given VM.
func getPrimaryInterfaceIDForVmssMachine(machine compute.VirtualMachineScaleSetVM) (string, error) {
	if len(*machine.NetworkProfile.NetworkInterfaces) == 1 {
		return *(*machine.NetworkProfile.NetworkInterfaces)[0].ID, nil
	}

	for _, ref := range *machine.NetworkProfile.NetworkInterfaces {
		if *ref.Primary {
			return *ref.ID, nil
		}
	}

	return "", fmt.Errorf("failed to find a primary nic for the vm. vmname=%q", *machine.Name)
}

// machineName is composed of computerNamePrefix and 36-based instanceID.
// And instanceID part if in fixed length of 6 characters.
// Refer https://msftstack.wordpress.com/2017/05/10/figuring-out-azure-vm-scale-set-machine-names/.
func getVmssInstanceID(machineName string) (string, error) {
	nameLength := len(machineName)
	if nameLength < 6 {
		return "", ErrorNotVmssInstance
	}

	instanceID, err := strconv.ParseUint(machineName[nameLength-6:], 36, 64)
	if err != nil {
		return "", ErrorNotVmssInstance
	}

	return fmt.Sprintf("%d", instanceID), nil
}
