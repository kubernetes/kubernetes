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
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/types"
	kwait "k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/cloudprovider"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/golang/glog"
)

const (
	defaultDataDiskCount       int = 16 // which will allow you to work with most medium size VMs (if not found in map)
	storageAccountNameTemplate     = "pvc%s"

	// for limits check https://docs.microsoft.com/en-us/azure/azure-subscription-service-limits#storage-limits
	maxStorageAccounts                     = 100 // max # is 200 (250 with special request). this allows 100 for everything else including stand alone disks
	maxDisksPerStorageAccounts             = 60
	storageAccountUtilizationBeforeGrowing = 0.5
	storageAccountsCountInit               = 2 // When the plug-in is init-ed, 2 storage accounts will be created to allow fast pvc create/attach/mount

	maxLUN               = 64 // max number of LUNs per VM
	errLeaseFailed       = "AcquireDiskLeaseFailed"
	errLeaseIDMissing    = "LeaseIdMissing"
	errContainerNotFound = "ContainerNotFound"
)

var defaultBackOff = kwait.Backoff{
	Steps:    20,
	Duration: 2 * time.Second,
	Factor:   1.5,
	Jitter:   0.0,
}

type controllerCommon struct {
	tenantID              string
	subscriptionID        string
	location              string
	storageEndpointSuffix string
	resourceGroup         string
	clientID              string
	clientSecret          string
	managementEndpoint    string
	tokenEndPoint         string
	aadResourceEndPoint   string
	aadToken              string
	expiresOn             time.Time
	cloud                 *Cloud
}

// AttachDisk attaches a vhd to vm
// the vhd must exist, can be identified by diskName, diskURI, and lun.
func (c *controllerCommon) AttachDisk(isManagedDisk bool, diskName, diskURI string, nodeName types.NodeName, lun int32, cachingMode compute.CachingTypes) error {
	vm, exists, err := c.cloud.getVirtualMachine(nodeName)
	if err != nil {
		return err
	} else if !exists {
		return cloudprovider.InstanceNotFound
	}
	disks := *vm.StorageProfile.DataDisks
	if isManagedDisk {
		disks = append(disks,
			compute.DataDisk{
				Name:         &diskName,
				Lun:          &lun,
				Caching:      cachingMode,
				CreateOption: "attach",
				ManagedDisk: &compute.ManagedDiskParameters{
					ID: &diskURI,
				},
			})
	} else {
		disks = append(disks,
			compute.DataDisk{
				Name: &diskName,
				Vhd: &compute.VirtualHardDisk{
					URI: &diskURI,
				},
				Lun:          &lun,
				Caching:      cachingMode,
				CreateOption: "attach",
			})
	}

	newVM := compute.VirtualMachine{
		Location: vm.Location,
		VirtualMachineProperties: &compute.VirtualMachineProperties{
			StorageProfile: &compute.StorageProfile{
				DataDisks: &disks,
			},
		},
	}
	vmName := mapNodeNameToVMName(nodeName)
	glog.V(2).Infof("azureDisk - update(%s): vm(%s) - attach disk", c.resourceGroup, vmName)
	c.cloud.operationPollRateLimiter.Accept()
	respChan, errChan := c.cloud.VirtualMachinesClient.CreateOrUpdate(c.resourceGroup, vmName, newVM, nil)
	resp := <-respChan
	err = <-errChan
	if c.cloud.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
		glog.V(2).Infof("azureDisk - update(%s) backing off: vm(%s)", c.resourceGroup, vmName)
		retryErr := c.cloud.CreateOrUpdateVMWithRetry(vmName, newVM)
		if retryErr != nil {
			err = retryErr
			glog.V(2).Infof("azureDisk - update(%s) abort backoff: vm(%s)", c.resourceGroup, vmName)
		}
	}
	if err != nil {
		glog.Errorf("azureDisk - azure attach failed, err: %v", err)
		detail := err.Error()
		if strings.Contains(detail, errLeaseFailed) {
			// if lease cannot be acquired, immediately detach the disk and return the original error
			glog.Infof("azureDisk - failed to acquire disk lease, try detach")
			c.cloud.DetachDiskByName(diskName, diskURI, nodeName)
		}
	} else {
		glog.V(4).Infof("azureDisk - azure attach succeeded")
	}
	return err
}

// DetachDiskByName detaches a vhd from host
// the vhd can be identified by diskName or diskURI
func (c *controllerCommon) DetachDiskByName(diskName, diskURI string, nodeName types.NodeName) error {
	vm, exists, err := c.cloud.getVirtualMachine(nodeName)
	if err != nil || !exists {
		// if host doesn't exist, no need to detach
		glog.Warningf("azureDisk - cannot find node %s, skip detaching disk %s", nodeName, diskName)
		return nil
	}

	disks := *vm.StorageProfile.DataDisks
	bFoundDisk := false
	for i, disk := range disks {
		if disk.Lun != nil && (disk.Name != nil && diskName != "" && *disk.Name == diskName) ||
			(disk.Vhd != nil && disk.Vhd.URI != nil && diskURI != "" && *disk.Vhd.URI == diskURI) ||
			(disk.ManagedDisk != nil && diskURI != "" && *disk.ManagedDisk.ID == diskURI) {
			// found the disk
			glog.V(4).Infof("azureDisk - detach disk: name %q uri %q", diskName, diskURI)
			disks = append(disks[:i], disks[i+1:]...)
			bFoundDisk = true
			break
		}
	}

	if !bFoundDisk {
		return fmt.Errorf("detach azure disk failure, disk %s not found, diskURI: %s", diskName, diskURI)
	}

	newVM := compute.VirtualMachine{
		Location: vm.Location,
		VirtualMachineProperties: &compute.VirtualMachineProperties{
			StorageProfile: &compute.StorageProfile{
				DataDisks: &disks,
			},
		},
	}
	vmName := mapNodeNameToVMName(nodeName)
	glog.V(2).Infof("azureDisk - update(%s): vm(%s) - detach disk", c.resourceGroup, vmName)
	c.cloud.operationPollRateLimiter.Accept()
	respChan, errChan := c.cloud.VirtualMachinesClient.CreateOrUpdate(c.resourceGroup, vmName, newVM, nil)
	resp := <-respChan
	err = <-errChan
	if c.cloud.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
		glog.V(2).Infof("azureDisk - update(%s) backing off: vm(%s)", c.resourceGroup, vmName)
		retryErr := c.cloud.CreateOrUpdateVMWithRetry(vmName, newVM)
		if retryErr != nil {
			err = retryErr
			glog.V(2).Infof("azureDisk - update(%s) abort backoff: vm(%s)", c.cloud.ResourceGroup, vmName)
		}
	}
	if err != nil {
		glog.Errorf("azureDisk - azure disk detach failed, err: %v", err)
	} else {
		glog.V(4).Infof("azureDisk - azure disk detach succeeded")
	}
	return err
}

// GetDiskLun finds the lun on the host that the vhd is attached to, given a vhd's diskName and diskURI
func (c *controllerCommon) GetDiskLun(diskName, diskURI string, nodeName types.NodeName) (int32, error) {
	vm, exists, err := c.cloud.getVirtualMachine(nodeName)
	if err != nil {
		return -1, err
	} else if !exists {
		return -1, cloudprovider.InstanceNotFound
	}
	disks := *vm.StorageProfile.DataDisks
	for _, disk := range disks {
		if disk.Lun != nil && (disk.Name != nil && diskName != "" && *disk.Name == diskName) ||
			(disk.Vhd != nil && disk.Vhd.URI != nil && diskURI != "" && *disk.Vhd.URI == diskURI) ||
			(disk.ManagedDisk != nil && *disk.ManagedDisk.ID == diskURI) {
			// found the disk
			glog.V(4).Infof("azureDisk - find disk: lun %d name %q uri %q", *disk.Lun, diskName, diskURI)
			return *disk.Lun, nil
		}
	}
	return -1, fmt.Errorf("Cannot find Lun for disk %s", diskName)
}

// GetNextDiskLun searches all vhd attachment on the host and find unused lun
// return -1 if all luns are used
func (c *controllerCommon) GetNextDiskLun(nodeName types.NodeName) (int32, error) {
	vm, exists, err := c.cloud.getVirtualMachine(nodeName)
	if err != nil {
		return -1, err
	} else if !exists {
		return -1, cloudprovider.InstanceNotFound
	}
	used := make([]bool, maxLUN)
	disks := *vm.StorageProfile.DataDisks
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
	return -1, fmt.Errorf("All Luns are used")
}

// DisksAreAttached checks if a list of volumes are attached to the node with the specified NodeName
func (c *controllerCommon) DisksAreAttached(diskNames []string, nodeName types.NodeName) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, diskName := range diskNames {
		attached[diskName] = false
	}
	vm, exists, err := c.cloud.getVirtualMachine(nodeName)
	if !exists {
		// if host doesn't exist, no need to detach
		glog.Warningf("azureDisk - Cannot find node %q, DisksAreAttached will assume disks %v are not attached to it.",
			nodeName, diskNames)
		return attached, nil
	} else if err != nil {
		return attached, err
	}

	disks := *vm.StorageProfile.DataDisks
	for _, disk := range disks {
		for _, diskName := range diskNames {
			if disk.Name != nil && diskName != "" && *disk.Name == diskName {
				attached[diskName] = true
			}
		}
	}

	return attached, nil
}
