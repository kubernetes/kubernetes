/*
Copyright 2018 The Kubernetes Authors.

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
	"time"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-04-01/compute"
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/types"
	kwait "k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

const (
	storageAccountNameTemplate = "pvc%s"

	// for limits check https://docs.microsoft.com/en-us/azure/azure-subscription-service-limits#storage-limits
	maxStorageAccounts                     = 100 // max # is 200 (250 with special request). this allows 100 for everything else including stand alone disks
	maxDisksPerStorageAccounts             = 60
	storageAccountUtilizationBeforeGrowing = 0.5

	maxLUN               = 64 // max number of LUNs per VM
	errLeaseFailed       = "AcquireDiskLeaseFailed"
	errLeaseIDMissing    = "LeaseIdMissing"
	errContainerNotFound = "ContainerNotFound"
	errDiskBlobNotFound  = "DiskBlobNotFound"
)

var defaultBackOff = kwait.Backoff{
	Steps:    20,
	Duration: 2 * time.Second,
	Factor:   1.5,
	Jitter:   0.0,
}

type controllerCommon struct {
	subscriptionID        string
	location              string
	storageEndpointSuffix string
	resourceGroup         string
	cloud                 *Cloud
}

// getNodeVMSet gets the VMSet interface based on config.VMType and the real virtual machine type.
func (c *controllerCommon) getNodeVMSet(nodeName types.NodeName) (VMSet, error) {
	// 1. vmType is standard, return cloud.vmSet directly.
	if c.cloud.VMType == vmTypeStandard {
		return c.cloud.vmSet, nil
	}

	// 2. vmType is Virtual Machine Scale Set (vmss), convert vmSet to scaleSet.
	ss, ok := c.cloud.vmSet.(*scaleSet)
	if !ok {
		return nil, fmt.Errorf("error of converting vmSet (%q) to scaleSet with vmType %q", c.cloud.vmSet, c.cloud.VMType)
	}

	// 3. If the node is managed by availability set, then return ss.availabilitySet.
	managedByAS, err := ss.isNodeManagedByAvailabilitySet(mapNodeNameToVMName(nodeName))
	if err != nil {
		return nil, err
	}
	if managedByAS {
		// vm is managed by availability set.
		return ss.availabilitySet, nil
	}

	// 4. Node is managed by vmss
	return ss, nil
}

// AttachDisk attaches a vhd to vm. The vhd must exist, can be identified by diskName, diskURI, and lun.
func (c *controllerCommon) AttachDisk(isManagedDisk bool, diskName, diskURI string, nodeName types.NodeName, lun int32, cachingMode compute.CachingTypes) error {
	vmset, err := c.getNodeVMSet(nodeName)
	if err != nil {
		return err
	}

	return vmset.AttachDisk(isManagedDisk, diskName, diskURI, nodeName, lun, cachingMode)
}

// DetachDiskByName detaches a vhd from host. The vhd can be identified by diskName or diskURI.
func (c *controllerCommon) DetachDiskByName(diskName, diskURI string, nodeName types.NodeName) error {
	vmset, err := c.getNodeVMSet(nodeName)
	if err != nil {
		return err
	}

	return vmset.DetachDiskByName(diskName, diskURI, nodeName)
}

// getNodeDataDisks invokes vmSet interfaces to get data disks for the node.
func (c *controllerCommon) getNodeDataDisks(nodeName types.NodeName) ([]compute.DataDisk, error) {
	vmset, err := c.getNodeVMSet(nodeName)
	if err != nil {
		return nil, err
	}

	return vmset.GetDataDisks(nodeName)
}

// GetDiskLun finds the lun on the host that the vhd is attached to, given a vhd's diskName and diskURI.
func (c *controllerCommon) GetDiskLun(diskName, diskURI string, nodeName types.NodeName) (int32, error) {
	disks, err := c.getNodeDataDisks(nodeName)
	if err != nil {
		glog.Errorf("error of getting data disks for node %q: %v", nodeName, err)
		return -1, err
	}

	for _, disk := range disks {
		if disk.Lun != nil && (disk.Name != nil && diskName != "" && *disk.Name == diskName) ||
			(disk.Vhd != nil && disk.Vhd.URI != nil && diskURI != "" && *disk.Vhd.URI == diskURI) ||
			(disk.ManagedDisk != nil && *disk.ManagedDisk.ID == diskURI) {
			// found the disk
			glog.V(2).Infof("azureDisk - find disk: lun %d name %q uri %q", *disk.Lun, diskName, diskURI)
			return *disk.Lun, nil
		}
	}
	return -1, fmt.Errorf("Cannot find Lun for disk %s", diskName)
}

// GetNextDiskLun searches all vhd attachment on the host and find unused lun. Return -1 if all luns are used.
func (c *controllerCommon) GetNextDiskLun(nodeName types.NodeName) (int32, error) {
	disks, err := c.getNodeDataDisks(nodeName)
	if err != nil {
		glog.Errorf("error of getting data disks for node %q: %v", nodeName, err)
		return -1, err
	}

	used := make([]bool, maxLUN)
	for _, disk := range disks {
		if disk.Lun != nil {
			used[*disk.Lun] = true
		}
	}
	for k, v := range used {
		if !v {
			return int32(k), nil
		}
	}
	return -1, fmt.Errorf("all luns are used")
}

// DisksAreAttached checks if a list of volumes are attached to the node with the specified NodeName.
func (c *controllerCommon) DisksAreAttached(diskNames []string, nodeName types.NodeName) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, diskName := range diskNames {
		attached[diskName] = false
	}

	disks, err := c.getNodeDataDisks(nodeName)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			// if host doesn't exist, no need to detach
			glog.Warningf("azureDisk - Cannot find node %q, DisksAreAttached will assume disks %v are not attached to it.",
				nodeName, diskNames)
			return attached, nil
		}

		return attached, err
	}

	for _, disk := range disks {
		for _, diskName := range diskNames {
			if disk.Name != nil && diskName != "" && *disk.Name == diskName {
				attached[diskName] = true
			}
		}
	}

	return attached, nil
}
