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

	"github.com/Azure/azure-sdk-for-go/arm/disk"
	storage "github.com/Azure/azure-sdk-for-go/arm/storage"
	"github.com/golang/glog"
	kwait "k8s.io/apimachinery/pkg/util/wait"
)

//ManagedDiskController : managed disk controller struct
type ManagedDiskController struct {
	common *controllerCommon
}

func newManagedDiskController(common *controllerCommon) (*ManagedDiskController, error) {
	return &ManagedDiskController{common: common}, nil
}

//CreateManagedDisk : create managed disk
func (c *ManagedDiskController) CreateManagedDisk(diskName string, storageAccountType storage.SkuName, resourceGroup string,
	sizeGB int, tags map[string]string) (string, error) {
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
	model := disk.Model{
		Location: &c.common.location,
		Tags:     &newTags,
		Properties: &disk.Properties{
			AccountType:  disk.StorageAccountTypes(storageAccountType),
			DiskSizeGB:   &diskSizeGB,
			CreationData: &disk.CreationData{CreateOption: disk.Empty},
		}}
	if resourceGroup == "" {
		resourceGroup = c.common.resourceGroup
	}
	cancel := make(chan struct{})
	respChan, errChan := c.common.cloud.DisksClient.CreateOrUpdate(resourceGroup, diskName, model, cancel)
	<-respChan
	err := <-errChan
	if err != nil {
		return "", err
	}

	diskID := ""

	err = kwait.ExponentialBackoff(defaultBackOff, func() (bool, error) {
		provisionState, id, err := c.getDisk(resourceGroup, diskName)
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
	resourceGroup, err := getResourceGroupFromDiskURI(diskURI)
	if err != nil {
		return err
	}
	cancel := make(chan struct{})
	respChan, errChan := c.common.cloud.DisksClient.Delete(resourceGroup, diskName, cancel)
	<-respChan
	err = <-errChan
	if err != nil {
		return err
	}
	// We don't need poll here, k8s will immediately stop referencing the disk
	// the disk will be eventually deleted - cleanly - by ARM

	glog.V(2).Infof("azureDisk - deleted a managed disk: %s", diskURI)

	return nil
}

// return: disk provisionState, diskID, error
func (c *ManagedDiskController) getDisk(resourceGroup, diskName string) (string, string, error) {
	result, err := c.common.cloud.DisksClient.Get(resourceGroup, diskName)
	if err != nil {
		return "", "", err
	}

	if result.Properties != nil && (*result.Properties).ProvisioningState != nil {
		return *(*result.Properties).ProvisioningState, *result.ID, nil
	}

	return "", "", err
}

// get resource group name from a managed disk URI, e.g. return {group-name} according to
// /subscriptions/{sub-id}/resourcegroups/{group-name}/providers/microsoft.compute/disks/{disk-id}
// according to https://docs.microsoft.com/en-us/rest/api/compute/disks/get
func getResourceGroupFromDiskURI(diskURI string) (string, error) {
	fields := strings.Split(diskURI, "/")
	if len(fields) != 9 || fields[3] != "resourceGroups" {
		return "", fmt.Errorf("invalid disk URI: %s", diskURI)
	}
	return fields[4], nil
}
