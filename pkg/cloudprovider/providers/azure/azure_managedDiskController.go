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
	"path"
	"strings"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2017-12-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2017-10-01/storage"
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/resource"
	kwait "k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume/util"
)

//ManagedDiskController : managed disk controller struct
type ManagedDiskController struct {
	common *controllerCommon
}

func newManagedDiskController(common *controllerCommon) (*ManagedDiskController, error) {
	return &ManagedDiskController{common: common}, nil
}

//CreateManagedDisk : create managed disk
func (c *ManagedDiskController) CreateManagedDisk(diskName string, storageAccountType storage.SkuName, sizeGB int, tags map[string]string) (string, error) {
	glog.V(4).Infof("azureDisk - creating new managed Name:%s StorageAccountType:%s Size:%v", diskName, storageAccountType, sizeGB)

	newTags := make(map[string]*string)
	azureDDTag := "kubernetes-azure-dd"
	newTags["created-by"] = &azureDDTag

	// insert original tags to newTags
	if tags != nil {
		for k, v := range tags {
			// Azure won't allow / (forward slash) in tags
			newKey := strings.Replace(k, "/", "-", -1)
			newValue := strings.Replace(v, "/", "-", -1)
			newTags[newKey] = &newValue
		}
	}

	diskSizeGB := int32(sizeGB)
	model := compute.Disk{
		Location: &c.common.location,
		Tags:     newTags,
		Sku: &compute.DiskSku{
			Name: compute.StorageAccountTypes(storageAccountType),
		},
		DiskProperties: &compute.DiskProperties{
			DiskSizeGB:   &diskSizeGB,
			CreationData: &compute.CreationData{CreateOption: compute.Empty},
		}}
	ctx, cancel := getContextWithCancel()
	defer cancel()
	_, err := c.common.cloud.DisksClient.CreateOrUpdate(ctx, c.common.resourceGroup, diskName, model)
	if err != nil {
		return "", err
	}

	diskID := ""

	err = kwait.ExponentialBackoff(defaultBackOff, func() (bool, error) {
		provisionState, id, err := c.getDisk(diskName)
		diskID = id
		// We are waiting for provisioningState==Succeeded
		// We don't want to hand-off managed disks to k8s while they are
		//still being provisioned, this is to avoid some race conditions
		if err != nil {
			return false, err
		}
		if strings.ToLower(provisionState) == "succeeded" {
			return true, nil
		}
		return false, nil
	})

	if err != nil {
		glog.V(2).Infof("azureDisk - created new MD Name:%s StorageAccountType:%s Size:%v but was unable to confirm provisioningState in poll process", diskName, storageAccountType, sizeGB)
	} else {
		glog.V(2).Infof("azureDisk - created new MD Name:%s StorageAccountType:%s Size:%v", diskName, storageAccountType, sizeGB)
	}

	return diskID, nil
}

//DeleteManagedDisk : delete managed disk
func (c *ManagedDiskController) DeleteManagedDisk(diskURI string) error {
	diskName := path.Base(diskURI)
	ctx, cancel := getContextWithCancel()
	defer cancel()

	_, err := c.common.cloud.DisksClient.Delete(ctx, c.common.resourceGroup, diskName)
	if err != nil {
		return err
	}
	// We don't need poll here, k8s will immediately stop referencing the disk
	// the disk will be eventually deleted - cleanly - by ARM

	glog.V(2).Infof("azureDisk - deleted a managed disk: %s", diskURI)

	return nil
}

// return: disk provisionState, diskID, error
func (c *ManagedDiskController) getDisk(diskName string) (string, string, error) {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	result, err := c.common.cloud.DisksClient.Get(ctx, c.common.resourceGroup, diskName)
	if err != nil {
		return "", "", err
	}

	if result.DiskProperties != nil && (*result.DiskProperties).ProvisioningState != nil {
		return *(*result.DiskProperties).ProvisioningState, *result.ID, nil
	}

	return "", "", err
}

// ResizeDisk Expand the disk to new size
func (c *ManagedDiskController) ResizeDisk(diskName string, oldSize resource.Quantity, newSize resource.Quantity) (resource.Quantity, error) {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	result, err := c.common.cloud.DisksClient.Get(ctx, c.common.resourceGroup, diskName)
	if err != nil {
		return oldSize, err
	}

	if result.DiskProperties == nil || result.DiskProperties.DiskSizeGB == nil {
		return oldSize, fmt.Errorf("DiskProperties of disk(%s) is nil", diskName)
	}

	requestBytes := newSize.Value()
	// Azure resizes in chunks of GiB (not GB)
	requestGiB := int32(util.RoundUpSize(requestBytes, 1024*1024*1024))
	newSizeQuant := resource.MustParse(fmt.Sprintf("%dGi", requestGiB))

	glog.V(2).Infof("azureDisk - begin to resize disk(%s) with new size(%d), old size(%v)", diskName, requestGiB, oldSize)
	// If disk already of greater or equal size than requested we return
	if *result.DiskProperties.DiskSizeGB >= requestGiB {
		return newSizeQuant, nil
	}

	result.DiskProperties.DiskSizeGB = &requestGiB

	ctx, cancel = getContextWithCancel()
	defer cancel()
	if _, err := c.common.cloud.DisksClient.CreateOrUpdate(ctx, c.common.resourceGroup, diskName, result); err != nil {
		return oldSize, err
	}

	glog.V(2).Infof("azureDisk - resize disk(%s) with new size(%d) completed", diskName, requestGiB)

	return newSizeQuant, nil
}
