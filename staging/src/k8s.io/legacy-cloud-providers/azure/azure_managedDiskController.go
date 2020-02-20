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
	"context"
	"fmt"
	"path"
	"strconv"
	"strings"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-03-01/compute"
	"github.com/Azure/go-autorest/autorest/to"
	"k8s.io/klog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	kwait "k8s.io/apimachinery/pkg/util/wait"
	cloudvolume "k8s.io/cloud-provider/volume"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"
)

const (
	// default IOPS Caps & Throughput Cap (MBps) per https://docs.microsoft.com/en-us/azure/virtual-machines/linux/disks-ultra-ssd
	defaultDiskIOPSReadWrite = 500
	defaultDiskMBpsReadWrite = 100
)

//ManagedDiskController : managed disk controller struct
type ManagedDiskController struct {
	common *controllerCommon
}

// ManagedDiskOptions specifies the options of managed disks.
type ManagedDiskOptions struct {
	// The name of the disk.
	DiskName string
	// The size in GB.
	SizeGB int
	// The name of PVC.
	PVCName string
	// The name of resource group.
	ResourceGroup string
	// The AvailabilityZone to create the disk.
	AvailabilityZone string
	// The tags of the disk.
	Tags map[string]string
	// The SKU of storage account.
	StorageAccountType compute.DiskStorageAccountTypes
	// IOPS Caps for UltraSSD disk
	DiskIOPSReadWrite string
	// Throughput Cap (MBps) for UltraSSD disk
	DiskMBpsReadWrite string
}

//CreateManagedDisk : create managed disk
func (c *ManagedDiskController) CreateManagedDisk(options *ManagedDiskOptions) (string, error) {
	var err error
	klog.V(4).Infof("azureDisk - creating new managed Name:%s StorageAccountType:%s Size:%v", options.DiskName, options.StorageAccountType, options.SizeGB)

	var createZones *[]string
	if len(options.AvailabilityZone) > 0 {
		zoneList := []string{c.common.cloud.GetZoneID(options.AvailabilityZone)}
		createZones = &zoneList
	}

	// insert original tags to newTags
	newTags := make(map[string]*string)
	azureDDTag := "kubernetes-azure-dd"
	newTags["created-by"] = &azureDDTag
	if options.Tags != nil {
		for k, v := range options.Tags {
			// Azure won't allow / (forward slash) in tags
			newKey := strings.Replace(k, "/", "-", -1)
			newValue := strings.Replace(v, "/", "-", -1)
			newTags[newKey] = &newValue
		}
	}

	diskSizeGB := int32(options.SizeGB)
	diskSku := compute.DiskStorageAccountTypes(options.StorageAccountType)
	diskProperties := compute.DiskProperties{
		DiskSizeGB:   &diskSizeGB,
		CreationData: &compute.CreationData{CreateOption: compute.Empty},
	}

	if diskSku == compute.UltraSSDLRS {
		diskIOPSReadWrite := int64(defaultDiskIOPSReadWrite)
		if options.DiskIOPSReadWrite != "" {
			v, err := strconv.Atoi(options.DiskIOPSReadWrite)
			if err != nil {
				return "", fmt.Errorf("AzureDisk - failed to parse DiskIOPSReadWrite: %v", err)
			}
			diskIOPSReadWrite = int64(v)
		}
		diskProperties.DiskIOPSReadWrite = to.Int64Ptr(diskIOPSReadWrite)

		diskMBpsReadWrite := int32(defaultDiskMBpsReadWrite)
		if options.DiskMBpsReadWrite != "" {
			v, err := strconv.Atoi(options.DiskMBpsReadWrite)
			if err != nil {
				return "", fmt.Errorf("AzureDisk - failed to parse DiskMBpsReadWrite: %v", err)
			}
			diskMBpsReadWrite = int32(v)
		}
		diskProperties.DiskMBpsReadWrite = to.Int32Ptr(diskMBpsReadWrite)
	} else {
		if options.DiskIOPSReadWrite != "" {
			return "", fmt.Errorf("AzureDisk - DiskIOPSReadWrite parameter is only applicable in UltraSSD_LRS disk type")
		}
		if options.DiskMBpsReadWrite != "" {
			return "", fmt.Errorf("AzureDisk - DiskMBpsReadWrite parameter is only applicable in UltraSSD_LRS disk type")
		}
	}

	model := compute.Disk{
		Location: &c.common.location,
		Tags:     newTags,
		Zones:    createZones,
		Sku: &compute.DiskSku{
			Name: diskSku,
		},
		DiskProperties: &diskProperties,
	}

	if options.ResourceGroup == "" {
		options.ResourceGroup = c.common.resourceGroup
	}

	ctx, cancel := getContextWithCancel()
	defer cancel()
	_, err = c.common.cloud.DisksClient.CreateOrUpdate(ctx, options.ResourceGroup, options.DiskName, model)
	if err != nil {
		return "", err
	}

	diskID := ""

	err = kwait.ExponentialBackoff(defaultBackOff, func() (bool, error) {
		provisionState, id, err := c.GetDisk(options.ResourceGroup, options.DiskName)
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
		klog.V(2).Infof("azureDisk - created new MD Name:%s StorageAccountType:%s Size:%v but was unable to confirm provisioningState in poll process", options.DiskName, options.StorageAccountType, options.SizeGB)
	} else {
		klog.V(2).Infof("azureDisk - created new MD Name:%s StorageAccountType:%s Size:%v", options.DiskName, options.StorageAccountType, options.SizeGB)
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

	ctx, cancel := getContextWithCancel()
	defer cancel()

	if _, ok := c.common.diskAttachDetachMap.Load(strings.ToLower(diskURI)); ok {
		return fmt.Errorf("failed to delete disk(%s) since it's in attaching or detaching state", diskURI)
	}

	disk, err := c.common.cloud.DisksClient.Get(ctx, resourceGroup, diskName)
	if err != nil {
		return err
	}

	if disk.ManagedBy != nil {
		return fmt.Errorf("disk(%s) already attached to node(%s), could not be deleted", diskURI, *disk.ManagedBy)
	}

	_, err = c.common.cloud.DisksClient.Delete(ctx, resourceGroup, diskName)
	if err != nil {
		return err
	}
	// We don't need poll here, k8s will immediately stop referencing the disk
	// the disk will be eventually deleted - cleanly - by ARM

	klog.V(2).Infof("azureDisk - deleted a managed disk: %s", diskURI)

	return nil
}

// GetDisk return: disk provisionState, diskID, error
func (c *ManagedDiskController) GetDisk(resourceGroup, diskName string) (string, string, error) {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	result, err := c.common.cloud.DisksClient.Get(ctx, resourceGroup, diskName)
	if err != nil {
		return "", "", err
	}

	if result.DiskProperties != nil && (*result.DiskProperties).ProvisioningState != nil {
		return *(*result.DiskProperties).ProvisioningState, *result.ID, nil
	}

	return "", "", err
}

// ResizeDisk Expand the disk to new size
func (c *ManagedDiskController) ResizeDisk(diskURI string, oldSize resource.Quantity, newSize resource.Quantity) (resource.Quantity, error) {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	diskName := path.Base(diskURI)
	resourceGroup, err := getResourceGroupFromDiskURI(diskURI)
	if err != nil {
		return oldSize, err
	}

	result, err := c.common.cloud.DisksClient.Get(ctx, resourceGroup, diskName)
	if err != nil {
		return oldSize, err
	}

	if result.DiskProperties == nil || result.DiskProperties.DiskSizeGB == nil {
		return oldSize, fmt.Errorf("DiskProperties of disk(%s) is nil", diskName)
	}

	// Azure resizes in chunks of GiB (not GB)
	requestGiB := int32(volumehelpers.RoundUpToGiB(newSize))
	newSizeQuant := resource.MustParse(fmt.Sprintf("%dGi", requestGiB))

	klog.V(2).Infof("azureDisk - begin to resize disk(%s) with new size(%d), old size(%v)", diskName, requestGiB, oldSize)
	// If disk already of greater or equal size than requested we return
	if *result.DiskProperties.DiskSizeGB >= requestGiB {
		return newSizeQuant, nil
	}

	result.DiskProperties.DiskSizeGB = &requestGiB

	ctx, cancel = getContextWithCancel()
	defer cancel()
	if _, err := c.common.cloud.DisksClient.CreateOrUpdate(ctx, resourceGroup, diskName, result); err != nil {
		return oldSize, err
	}

	klog.V(2).Infof("azureDisk - resize disk(%s) with new size(%d) completed", diskName, requestGiB)

	return newSizeQuant, nil
}

// get resource group name from a managed disk URI, e.g. return {group-name} according to
// /subscriptions/{sub-id}/resourcegroups/{group-name}/providers/microsoft.compute/disks/{disk-id}
// according to https://docs.microsoft.com/en-us/rest/api/compute/disks/get
func getResourceGroupFromDiskURI(diskURI string) (string, error) {
	fields := strings.Split(diskURI, "/")
	if len(fields) != 9 || strings.ToLower(fields[3]) != "resourcegroups" {
		return "", fmt.Errorf("invalid disk URI: %s", diskURI)
	}
	return fields[4], nil
}

// GetLabelsForVolume implements PVLabeler.GetLabelsForVolume
func (c *Cloud) GetLabelsForVolume(ctx context.Context, pv *v1.PersistentVolume) (map[string]string, error) {
	// Ignore if not AzureDisk.
	if pv.Spec.AzureDisk == nil {
		return nil, nil
	}

	// Ignore any volumes that are being provisioned
	if pv.Spec.AzureDisk.DiskName == cloudvolume.ProvisionedVolumeName {
		return nil, nil
	}

	return c.GetAzureDiskLabels(pv.Spec.AzureDisk.DataDiskURI)
}

// GetAzureDiskLabels gets availability zone labels for Azuredisk.
func (c *Cloud) GetAzureDiskLabels(diskURI string) (map[string]string, error) {
	// Get disk's resource group.
	diskName := path.Base(diskURI)
	resourceGroup, err := getResourceGroupFromDiskURI(diskURI)
	if err != nil {
		klog.Errorf("Failed to get resource group for AzureDisk %q: %v", diskName, err)
		return nil, err
	}

	// Get information of the disk.
	ctx, cancel := getContextWithCancel()
	defer cancel()
	disk, err := c.DisksClient.Get(ctx, resourceGroup, diskName)
	if err != nil {
		klog.Errorf("Failed to get information for AzureDisk %q: %v", diskName, err)
		return nil, err
	}

	// Check whether availability zone is specified.
	if disk.Zones == nil || len(*disk.Zones) == 0 {
		klog.V(4).Infof("Azure disk %q is not zoned", diskName)
		return nil, nil
	}

	zones := *disk.Zones
	zoneID, err := strconv.Atoi(zones[0])
	if err != nil {
		return nil, fmt.Errorf("failed to parse zone %v for AzureDisk %v: %v", zones, diskName, err)
	}

	zone := c.makeZone(zoneID)
	klog.V(4).Infof("Got zone %q for Azure disk %q", zone, diskName)
	labels := map[string]string{
		v1.LabelZoneRegion:        c.Location,
		v1.LabelZoneFailureDomain: zone,
	}
	return labels, nil
}
