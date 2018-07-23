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

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-04-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2017-10-01/storage"
	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/sets"
	kwait "k8s.io/apimachinery/pkg/util/wait"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

//ManagedDiskController : managed disk controller struct
type ManagedDiskController struct {
	common *controllerCommon
}

// ManagedDiskOptions specifies the options of managed disks.
type ManagedDiskOptions struct {
	DiskName           string
	SizeGB             int
	PVCName            string
	ResourceGroup      string
	Zoned              bool
	ZonePresent        bool
	ZonesPresent       bool
	AvailabilityZone   string
	AvailabilityZones  string
	Tags               map[string]string
	StorageAccountType storage.SkuName
}

func newManagedDiskController(common *controllerCommon) (*ManagedDiskController, error) {
	return &ManagedDiskController{common: common}, nil
}

//CreateManagedDisk : create managed disk
func (c *ManagedDiskController) CreateManagedDisk(options *ManagedDiskOptions) (string, error) {
	var zones sets.String
	var activeZones sets.String
	var err error
	glog.V(4).Infof("azureDisk - creating new managed Name:%s StorageAccountType:%s Size:%v", options.DiskName, options.StorageAccountType, options.SizeGB)

	// Get active zones which have nodes running on.
	activeZones, err = c.common.cloud.GetActiveZones()
	if err != nil {
		return "", fmt.Errorf("error querying active zones: %v", err)
	}

	// Validate and choose availability zone for creating disk.
	if options.Zoned && !options.ZonePresent && !options.ZonesPresent {
		// Neither "zone" or "zones" specified. Pick a zone randomly selected
		// from all active zones where Kubernetes cluster has a node.
		zones = activeZones
	} else if !options.ZonePresent && options.ZonesPresent {
		// Choose zone from specified zones.
		if zones, err = util.ZonesToSet(options.AvailabilityZones); err != nil {
			return "", err
		}
	} else if options.ZonePresent && !options.ZonesPresent {
		if err := util.ValidateZone(options.AvailabilityZone); err != nil {
			return "", err
		}
		zones = make(sets.String)
		zones.Insert(options.AvailabilityZone)
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
	model := compute.Disk{
		Location: &c.common.location,
		Tags:     newTags,
		Sku: &compute.DiskSku{
			Name: compute.StorageAccountTypes(options.StorageAccountType),
		},
		DiskProperties: &compute.DiskProperties{
			DiskSizeGB:   &diskSizeGB,
			CreationData: &compute.CreationData{CreateOption: compute.Empty},
		},
	}

	if options.ResourceGroup == "" {
		options.ResourceGroup = c.common.resourceGroup
	}
	if len(zones.List()) > 0 {
		createAZ := util.ChooseZoneForVolume(zones, options.PVCName)
		// Do not allow creation of disks in zones that are do not have nodes. Such disks
		// are not currently usable.
		if !activeZones.Has(createAZ) {
			return "", fmt.Errorf("kubernetes does not have a node in zone %q", createAZ)
		}

		createZones := []string{c.common.cloud.GetZoneID(createAZ)}
		model.Zones = &createZones
	}

	ctx, cancel := getContextWithCancel()
	defer cancel()
	_, err = c.common.cloud.DisksClient.CreateOrUpdate(ctx, options.ResourceGroup, options.DiskName, model)
	if err != nil {
		return "", err
	}

	diskID := ""

	err = kwait.ExponentialBackoff(defaultBackOff, func() (bool, error) {
		provisionState, id, err := c.getDisk(options.ResourceGroup, options.DiskName)
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
		glog.V(2).Infof("azureDisk - created new MD Name:%s StorageAccountType:%s Size:%v but was unable to confirm provisioningState in poll process", options.DiskName, options.StorageAccountType, options.SizeGB)
	} else {
		glog.V(2).Infof("azureDisk - created new MD Name:%s StorageAccountType:%s Size:%v", options.DiskName, options.StorageAccountType, options.SizeGB)
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

	_, err = c.common.cloud.DisksClient.Delete(ctx, resourceGroup, diskName)
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
	if _, err := c.common.cloud.DisksClient.CreateOrUpdate(ctx, resourceGroup, diskName, result); err != nil {
		return oldSize, err
	}

	glog.V(2).Infof("azureDisk - resize disk(%s) with new size(%d) completed", diskName, requestGiB)

	return newSizeQuant, nil
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

// GetLabelsForVolume implements PVLabeler.GetLabelsForVolume
func (c *Cloud) GetLabelsForVolume(ctx context.Context, pv *v1.PersistentVolume) (map[string]string, error) {
	// Ignore if not AzureDisk.
	if pv.Spec.AzureDisk == nil {
		return nil, nil
	}

	// Ignore any volumes that are being provisioned
	if pv.Spec.AzureDisk.DiskName == volume.ProvisionedVolumeName {
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
		glog.Errorf("Failed to get resource group for AzureDisk %q: %v", diskName, err)
		return nil, err
	}

	// Get information of the disk.
	ctx, cancel := getContextWithCancel()
	defer cancel()
	disk, err := c.DisksClient.Get(ctx, resourceGroup, diskName)
	if err != nil {
		glog.Errorf("Failed to get information for AzureDisk %q: %v", diskName, err)
		return nil, err
	}

	// Check whether availability zone is specified.
	if disk.Zones == nil || len(*disk.Zones) == 0 {
		glog.V(4).Infof("Azure disk %q is not zoned", diskName)
		return nil, nil
	}

	zones := *disk.Zones
	zoneID, err := strconv.Atoi(zones[0])
	if err != nil {
		return nil, fmt.Errorf("failed to parse zone %v for AzureDisk %v: %v", zones, diskName, err)
	}

	zone := c.makeZone(zoneID)
	glog.V(4).Infof("Get zone %q for Azure disk %q", zone, diskName)
	labels := map[string]string{
		kubeletapis.LabelZoneRegion:        c.Location,
		kubeletapis.LabelZoneFailureDomain: zone,
	}
	return labels, nil
}
