// +build !providerless

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
	"context"
	"fmt"
	"net/http"
	"path"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"

	"k8s.io/apimachinery/pkg/types"
	kwait "k8s.io/apimachinery/pkg/util/wait"
	cloudprovider "k8s.io/cloud-provider"
	volerr "k8s.io/cloud-provider/volume/errors"
	"k8s.io/klog/v2"
	azcache "k8s.io/legacy-cloud-providers/azure/cache"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

const (
	// for limits check https://docs.microsoft.com/en-us/azure/azure-subscription-service-limits#storage-limits
	maxStorageAccounts                     = 100 // max # is 200 (250 with special request). this allows 100 for everything else including stand alone disks
	maxDisksPerStorageAccounts             = 60
	storageAccountUtilizationBeforeGrowing = 0.5
	// Disk Caching is not supported for disks 4 TiB and larger
	// https://docs.microsoft.com/en-us/azure/virtual-machines/premium-storage-performance#disk-caching
	diskCachingLimit = 4096 // GiB

	maxLUN                 = 64 // max number of LUNs per VM
	errLeaseFailed         = "AcquireDiskLeaseFailed"
	errLeaseIDMissing      = "LeaseIdMissing"
	errContainerNotFound   = "ContainerNotFound"
	errStatusCode400       = "statuscode=400"
	errInvalidParameter    = `code="invalidparameter"`
	errTargetInstanceIds   = `target="instanceids"`
	sourceSnapshot         = "snapshot"
	sourceVolume           = "volume"
	attachDiskMapKeySuffix = "attachdiskmap"
	detachDiskMapKeySuffix = "detachdiskmap"

	// WriteAcceleratorEnabled support for Azure Write Accelerator on Azure Disks
	// https://docs.microsoft.com/azure/virtual-machines/windows/how-to-enable-write-accelerator
	WriteAcceleratorEnabled = "writeacceleratorenabled"

	// see https://docs.microsoft.com/en-us/rest/api/compute/disks/createorupdate#create-a-managed-disk-by-copying-a-snapshot.
	diskSnapshotPath = "/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/snapshots/%s"

	// see https://docs.microsoft.com/en-us/rest/api/compute/disks/createorupdate#create-a-managed-disk-from-an-existing-managed-disk-in-the-same-or-different-subscription.
	managedDiskPath = "/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/disks/%s"
)

var defaultBackOff = kwait.Backoff{
	Steps:    20,
	Duration: 2 * time.Second,
	Factor:   1.5,
	Jitter:   0.0,
}

var (
	managedDiskPathRE  = regexp.MustCompile(`.*/subscriptions/(?:.*)/resourceGroups/(?:.*)/providers/Microsoft.Compute/disks/(.+)`)
	diskSnapshotPathRE = regexp.MustCompile(`.*/subscriptions/(?:.*)/resourceGroups/(?:.*)/providers/Microsoft.Compute/snapshots/(.+)`)
)

type controllerCommon struct {
	subscriptionID        string
	location              string
	storageEndpointSuffix string
	resourceGroup         string
	diskStateMap          sync.Map // <diskURI, attaching/detaching state>
	lockMap               *lockMap
	cloud                 *Cloud
	// disk queue that is waiting for attach or detach on specific node
	// <nodeName, map<diskURI, *AttachDiskOptions/DetachDiskOptions>>
	attachDiskMap sync.Map
	detachDiskMap sync.Map
}

// AttachDiskOptions attach disk options
type AttachDiskOptions struct {
	lun                     int32
	isManagedDisk           bool
	diskName                string
	cachingMode             compute.CachingTypes
	diskEncryptionSetID     string
	writeAcceleratorEnabled bool
}

// getNodeVMSet gets the VMSet interface based on config.VMType and the real virtual machine type.
func (c *controllerCommon) getNodeVMSet(nodeName types.NodeName, crt azcache.AzureCacheReadType) (VMSet, error) {
	// 1. vmType is standard, return cloud.VMSet directly.
	if c.cloud.VMType == vmTypeStandard {
		return c.cloud.VMSet, nil
	}

	// 2. vmType is Virtual Machine Scale Set (vmss), convert vmSet to scaleSet.
	ss, ok := c.cloud.VMSet.(*scaleSet)
	if !ok {
		return nil, fmt.Errorf("error of converting vmSet (%q) to scaleSet with vmType %q", c.cloud.VMSet, c.cloud.VMType)
	}

	// 3. If the node is managed by availability set, then return ss.availabilitySet.
	managedByAS, err := ss.isNodeManagedByAvailabilitySet(mapNodeNameToVMName(nodeName), crt)
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

// AttachDisk attaches a vhd to vm. The vhd must exist, can be identified by diskName, diskURI.
// return (lun, error)
func (c *controllerCommon) AttachDisk(isManagedDisk bool, diskName, diskURI string, nodeName types.NodeName, cachingMode compute.CachingTypes) (int32, error) {
	diskEncryptionSetID := ""
	writeAcceleratorEnabled := false

	vmset, err := c.getNodeVMSet(nodeName, azcache.CacheReadTypeUnsafe)
	if err != nil {
		return -1, err
	}

	if isManagedDisk {
		diskName := path.Base(diskURI)
		resourceGroup, err := getResourceGroupFromDiskURI(diskURI)
		if err != nil {
			return -1, err
		}

		ctx, cancel := getContextWithCancel()
		defer cancel()

		disk, rerr := c.cloud.DisksClient.Get(ctx, resourceGroup, diskName)
		if rerr != nil {
			return -1, rerr.Error()
		}

		if disk.ManagedBy != nil && (disk.MaxShares == nil || *disk.MaxShares <= 1) {
			attachErr := fmt.Sprintf(
				"disk(%s) already attached to node(%s), could not be attached to node(%s)",
				diskURI, *disk.ManagedBy, nodeName)
			attachedNode, err := vmset.GetNodeNameByProviderID(*disk.ManagedBy)
			if err != nil {
				return -1, err
			}
			klog.V(2).Infof("found dangling volume %s attached to node %s", diskURI, attachedNode)
			danglingErr := volerr.NewDanglingError(attachErr, attachedNode, "")
			return -1, danglingErr
		}

		if disk.DiskProperties != nil {
			if disk.DiskProperties.DiskSizeGB != nil && *disk.DiskProperties.DiskSizeGB >= diskCachingLimit && cachingMode != compute.CachingTypesNone {
				// Disk Caching is not supported for disks 4 TiB and larger
				// https://docs.microsoft.com/en-us/azure/virtual-machines/premium-storage-performance#disk-caching
				cachingMode = compute.CachingTypesNone
				klog.Warningf("size of disk(%s) is %dGB which is bigger than limit(%dGB), set cacheMode as None",
					diskURI, *disk.DiskProperties.DiskSizeGB, diskCachingLimit)
			}

			if disk.DiskProperties.Encryption != nil &&
				disk.DiskProperties.Encryption.DiskEncryptionSetID != nil {
				diskEncryptionSetID = *disk.DiskProperties.Encryption.DiskEncryptionSetID
			}
		}

		if v, ok := disk.Tags[WriteAcceleratorEnabled]; ok {
			if v != nil && strings.EqualFold(*v, "true") {
				writeAcceleratorEnabled = true
			}
		}
	}

	instanceid, err := c.cloud.InstanceID(context.TODO(), nodeName)
	if err != nil {
		klog.Warningf("failed to get azure instance id (%v) for node %s", err, nodeName)
		return -1, fmt.Errorf("failed to get azure instance id for node %q (%v)", nodeName, err)
	}

	options := AttachDiskOptions{
		lun:                     -1,
		isManagedDisk:           isManagedDisk,
		diskName:                diskName,
		cachingMode:             cachingMode,
		diskEncryptionSetID:     diskEncryptionSetID,
		writeAcceleratorEnabled: writeAcceleratorEnabled,
	}
	node := strings.ToLower(string(nodeName))
	disk := strings.ToLower(string(diskURI))
	if err := c.insertAttachDiskRequest(disk, node, &options); err != nil {
		return -1, err
	}

	c.lockMap.LockEntry(node)
	defer c.lockMap.UnlockEntry(node)
	diskMap, err := c.cleanAttachDiskRequests(node)
	if err != nil {
		return -1, err
	}

	lun, err := c.SetDiskLun(nodeName, disk, diskMap)
	if err != nil {
		klog.Warningf("no LUN available for instance %q (%v)", nodeName, err)
		return -1, fmt.Errorf("all LUNs are used, cannot attach volume (%s, %s) to instance %q (%v)", diskName, diskURI, instanceid, err)
	}

	klog.V(2).Infof("Trying to attach volume %q lun %d to node %q, diskMap: %s", diskURI, lun, nodeName, diskMap)
	if len(diskMap) == 0 {
		return lun, nil
	}
	c.diskStateMap.Store(disk, "attaching")
	defer c.diskStateMap.Delete(disk)
	return lun, vmset.AttachDisk(nodeName, diskMap)
}

func (c *controllerCommon) insertAttachDiskRequest(diskURI, nodeName string, options *AttachDiskOptions) error {
	var diskMap map[string]*AttachDiskOptions
	attachDiskMapKey := nodeName + attachDiskMapKeySuffix
	c.lockMap.LockEntry(attachDiskMapKey)
	defer c.lockMap.UnlockEntry(attachDiskMapKey)
	v, ok := c.attachDiskMap.Load(nodeName)
	if ok {
		if diskMap, ok = v.(map[string]*AttachDiskOptions); !ok {
			return fmt.Errorf("convert attachDiskMap failure on node(%s)", nodeName)
		}
	} else {
		diskMap = make(map[string]*AttachDiskOptions)
		c.attachDiskMap.Store(nodeName, diskMap)
	}
	// insert attach disk request to queue
	_, ok = diskMap[diskURI]
	if ok {
		klog.V(2).Infof("azureDisk - duplicated attach disk(%s) request on node(%s)", diskURI, nodeName)
	} else {
		diskMap[diskURI] = options
	}
	return nil
}

// clean up attach disk requests
// return original attach disk requests
func (c *controllerCommon) cleanAttachDiskRequests(nodeName string) (map[string]*AttachDiskOptions, error) {
	var diskMap map[string]*AttachDiskOptions

	attachDiskMapKey := nodeName + attachDiskMapKeySuffix
	c.lockMap.LockEntry(attachDiskMapKey)
	defer c.lockMap.UnlockEntry(attachDiskMapKey)
	v, ok := c.attachDiskMap.Load(nodeName)
	if !ok {
		return diskMap, nil
	}
	if diskMap, ok = v.(map[string]*AttachDiskOptions); !ok {
		return diskMap, fmt.Errorf("convert attachDiskMap failure on node(%s)", nodeName)
	}
	c.attachDiskMap.Store(nodeName, make(map[string]*AttachDiskOptions))
	return diskMap, nil
}

// DetachDisk detaches a disk from VM
func (c *controllerCommon) DetachDisk(diskName, diskURI string, nodeName types.NodeName) error {
	_, err := c.cloud.InstanceID(context.TODO(), nodeName)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			// if host doesn't exist, no need to detach
			klog.Warningf("azureDisk - failed to get azure instance id(%q), DetachDisk(%s) will assume disk is already detached",
				nodeName, diskURI)
			return nil
		}
		klog.Warningf("failed to get azure instance id (%v)", err)
		return fmt.Errorf("failed to get azure instance id for node %q (%v)", nodeName, err)
	}

	vmset, err := c.getNodeVMSet(nodeName, azcache.CacheReadTypeUnsafe)
	if err != nil {
		return err
	}

	node := strings.ToLower(string(nodeName))
	disk := strings.ToLower(string(diskURI))
	if err := c.insertDetachDiskRequest(diskName, disk, node); err != nil {
		return err
	}

	c.lockMap.LockEntry(node)
	diskMap, err := c.cleanDetachDiskRequests(node)
	if err != nil {
		return err
	}

	klog.V(2).Infof("Trying to detach volume %q from node %q, diskMap: %s", diskURI, nodeName, diskMap)
	if len(diskMap) > 0 {
		c.diskStateMap.Store(disk, "detaching")
		err = vmset.DetachDisk(nodeName, diskMap)
		c.diskStateMap.Delete(disk)
	}
	c.lockMap.UnlockEntry(node)

	if err != nil {
		if isInstanceNotFoundError(err) {
			// if host doesn't exist, no need to detach
			klog.Warningf("azureDisk - got InstanceNotFoundError(%v), DetachDisk(%s) will assume disk is already detached",
				err, diskURI)
			return nil
		}
		if retry.IsErrorRetriable(err) && c.cloud.CloudProviderBackoff {
			klog.Warningf("azureDisk - update backing off: detach disk(%s, %s), err: %v", diskName, diskURI, err)
			retryErr := kwait.ExponentialBackoff(c.cloud.RequestBackoff(), func() (bool, error) {
				c.lockMap.LockEntry(node)
				c.diskStateMap.Store(disk, "detaching")
				err := vmset.DetachDisk(nodeName, diskMap)
				c.diskStateMap.Delete(disk)
				c.lockMap.UnlockEntry(node)

				retriable := false
				if err != nil && retry.IsErrorRetriable(err) {
					retriable = true
				}
				return !retriable, err
			})
			if retryErr != nil {
				err = retryErr
				klog.V(2).Infof("azureDisk - update abort backoff: detach disk(%s, %s), err: %v", diskName, diskURI, err)
			}
		}
	}
	if err != nil {
		klog.Errorf("azureDisk - detach disk(%s, %s) failed, err: %v", diskName, diskURI, err)
		return err
	}

	klog.V(2).Infof("azureDisk - detach disk(%s, %s) succeeded", diskName, diskURI)
	return nil
}

func (c *controllerCommon) insertDetachDiskRequest(diskName, diskURI, nodeName string) error {
	var diskMap map[string]string
	detachDiskMapKey := nodeName + detachDiskMapKeySuffix
	c.lockMap.LockEntry(detachDiskMapKey)
	defer c.lockMap.UnlockEntry(detachDiskMapKey)
	v, ok := c.detachDiskMap.Load(nodeName)
	if ok {
		if diskMap, ok = v.(map[string]string); !ok {
			return fmt.Errorf("convert detachDiskMap failure on node(%s)", nodeName)
		}
	} else {
		diskMap = make(map[string]string)
		c.detachDiskMap.Store(nodeName, diskMap)
	}
	// insert detach disk request to queue
	_, ok = diskMap[diskURI]
	if ok {
		klog.V(2).Infof("azureDisk - duplicated detach disk(%s) request on node(%s)", diskURI, nodeName)
	} else {
		diskMap[diskURI] = diskName
	}
	return nil
}

// clean up detach disk requests
// return original detach disk requests
func (c *controllerCommon) cleanDetachDiskRequests(nodeName string) (map[string]string, error) {
	var diskMap map[string]string

	detachDiskMapKey := nodeName + detachDiskMapKeySuffix
	c.lockMap.LockEntry(detachDiskMapKey)
	defer c.lockMap.UnlockEntry(detachDiskMapKey)
	v, ok := c.detachDiskMap.Load(nodeName)
	if !ok {
		return diskMap, nil
	}
	if diskMap, ok = v.(map[string]string); !ok {
		return diskMap, fmt.Errorf("convert detachDiskMap failure on node(%s)", nodeName)
	}
	// clean up original requests in disk map
	c.detachDiskMap.Store(nodeName, make(map[string]string))
	return diskMap, nil
}

// getNodeDataDisks invokes vmSet interfaces to get data disks for the node.
func (c *controllerCommon) getNodeDataDisks(nodeName types.NodeName, crt azcache.AzureCacheReadType) ([]compute.DataDisk, error) {
	vmset, err := c.getNodeVMSet(nodeName, crt)
	if err != nil {
		return nil, err
	}

	return vmset.GetDataDisks(nodeName, crt)
}

// GetDiskLun finds the lun on the host that the vhd is attached to, given a vhd's diskName and diskURI.
func (c *controllerCommon) GetDiskLun(diskName, diskURI string, nodeName types.NodeName) (int32, error) {
	// getNodeDataDisks need to fetch the cached data/fresh data if cache expired here
	// to ensure we get LUN based on latest entry.
	disks, err := c.getNodeDataDisks(nodeName, azcache.CacheReadTypeDefault)
	if err != nil {
		klog.Errorf("error of getting data disks for node %q: %v", nodeName, err)
		return -1, err
	}

	for _, disk := range disks {
		if disk.Lun != nil && (disk.Name != nil && diskName != "" && strings.EqualFold(*disk.Name, diskName)) ||
			(disk.Vhd != nil && disk.Vhd.URI != nil && diskURI != "" && strings.EqualFold(*disk.Vhd.URI, diskURI)) ||
			(disk.ManagedDisk != nil && strings.EqualFold(*disk.ManagedDisk.ID, diskURI)) {
			if disk.ToBeDetached != nil && *disk.ToBeDetached {
				klog.Warningf("azureDisk - find disk(ToBeDetached): lun %d name %q uri %q", *disk.Lun, diskName, diskURI)
			} else {
				// found the disk
				klog.V(2).Infof("azureDisk - find disk: lun %d name %q uri %q", *disk.Lun, diskName, diskURI)
				return *disk.Lun, nil
			}
		}
	}
	return -1, fmt.Errorf("cannot find Lun for disk %s", diskName)
}

// SetDiskLun find unused luns and allocate lun for every disk in diskMap.
// Return lun of diskURI, -1 if all luns are used.
func (c *controllerCommon) SetDiskLun(nodeName types.NodeName, diskURI string, diskMap map[string]*AttachDiskOptions) (int32, error) {
	disks, err := c.getNodeDataDisks(nodeName, azcache.CacheReadTypeForceRefresh)
	if err != nil {
		klog.Errorf("error of getting data disks for node %q: %v", nodeName, err)
		return -1, err
	}

	lun := int32(-1)
	_, isDiskInMap := diskMap[diskURI]
	used := make([]bool, maxLUN)
	for _, disk := range disks {
		if disk.Lun != nil {
			used[*disk.Lun] = true
			if !isDiskInMap {
				// find lun of diskURI since diskURI is not in diskMap
				if disk.ManagedDisk != nil && strings.EqualFold(*disk.ManagedDisk.ID, diskURI) {
					lun = *disk.Lun
				}
			}
		}
	}
	if !isDiskInMap && lun < 0 {
		return -1, fmt.Errorf("could not find disk(%s) in current disk list(len: %d) nor in diskMap(%v)", diskURI, len(disks), diskMap)
	}
	if len(diskMap) == 0 {
		// attach disk request is empty, return directly
		return lun, nil
	}

	// allocate lun for every disk in diskMap
	var diskLuns []int32
	count := 0
	for k, v := range used {
		if !v {
			diskLuns = append(diskLuns, int32(k))
			count++
			if count >= len(diskMap) {
				break
			}
		}
	}

	if len(diskLuns) != len(diskMap) {
		return -1, fmt.Errorf("could not find enough disk luns(current: %d) for diskMap(%v, len=%d), diskURI(%s)",
			len(diskLuns), diskMap, len(diskMap), diskURI)
	}

	count = 0
	for uri, opt := range diskMap {
		if opt == nil {
			return -1, fmt.Errorf("unexpected nil pointer in diskMap(%v), diskURI(%s)", diskMap, diskURI)
		}
		if strings.EqualFold(uri, diskURI) {
			lun = diskLuns[count]
		}
		opt.lun = diskLuns[count]
		count++
	}
	if lun < 0 {
		return lun, fmt.Errorf("could not find lun of diskURI(%s), diskMap(%v)", diskURI, diskMap)
	}
	return lun, nil
}

// DisksAreAttached checks if a list of volumes are attached to the node with the specified NodeName.
func (c *controllerCommon) DisksAreAttached(diskNames []string, nodeName types.NodeName) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, diskName := range diskNames {
		attached[diskName] = false
	}

	// doing stalled read for getNodeDataDisks to ensure we don't call ARM
	// for every reconcile call. The cache is invalidated after Attach/Detach
	// disk. So the new entry will be fetched and cached the first time reconcile
	// loop runs after the Attach/Disk OP which will reflect the latest model.
	disks, err := c.getNodeDataDisks(nodeName, azcache.CacheReadTypeUnsafe)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			// if host doesn't exist, no need to detach
			klog.Warningf("azureDisk - Cannot find node %q, DisksAreAttached will assume disks %v are not attached to it.",
				nodeName, diskNames)
			return attached, nil
		}

		return attached, err
	}

	for _, disk := range disks {
		for _, diskName := range diskNames {
			if disk.Name != nil && diskName != "" && strings.EqualFold(*disk.Name, diskName) {
				attached[diskName] = true
			}
		}
	}

	return attached, nil
}

func filterDetachingDisks(unfilteredDisks []compute.DataDisk) []compute.DataDisk {
	filteredDisks := []compute.DataDisk{}
	for _, disk := range unfilteredDisks {
		if disk.ToBeDetached != nil && *disk.ToBeDetached {
			if disk.Name != nil {
				klog.V(2).Infof("Filtering disk: %s with ToBeDetached flag set.", *disk.Name)
			}
		} else {
			filteredDisks = append(filteredDisks, disk)
		}
	}
	return filteredDisks
}

func (c *controllerCommon) filterNonExistingDisks(ctx context.Context, unfilteredDisks []compute.DataDisk) []compute.DataDisk {
	filteredDisks := []compute.DataDisk{}
	for _, disk := range unfilteredDisks {
		filter := false
		if disk.ManagedDisk != nil && disk.ManagedDisk.ID != nil {
			diskURI := *disk.ManagedDisk.ID
			exist, err := c.cloud.checkDiskExists(ctx, diskURI)
			if err != nil {
				klog.Errorf("checkDiskExists(%s) failed with error: %v", diskURI, err)
			} else {
				// only filter disk when checkDiskExists returns <false, nil>
				filter = !exist
				if filter {
					klog.Errorf("disk(%s) does not exist, removed from data disk list", diskURI)
				}
			}
		}

		if !filter {
			filteredDisks = append(filteredDisks, disk)
		}
	}
	return filteredDisks
}

func (c *controllerCommon) checkDiskExists(ctx context.Context, diskURI string) (bool, error) {
	diskName := path.Base(diskURI)
	resourceGroup, err := getResourceGroupFromDiskURI(diskURI)
	if err != nil {
		return false, err
	}

	if _, rerr := c.cloud.DisksClient.Get(ctx, resourceGroup, diskName); rerr != nil {
		if rerr.HTTPStatusCode == http.StatusNotFound {
			return false, nil
		}
		return false, rerr.Error()
	}

	return true, nil
}

func getValidCreationData(subscriptionID, resourceGroup, sourceResourceID, sourceType string) (compute.CreationData, error) {
	if sourceResourceID == "" {
		return compute.CreationData{
			CreateOption: compute.Empty,
		}, nil
	}

	switch sourceType {
	case sourceSnapshot:
		if match := diskSnapshotPathRE.FindString(sourceResourceID); match == "" {
			sourceResourceID = fmt.Sprintf(diskSnapshotPath, subscriptionID, resourceGroup, sourceResourceID)
		}

	case sourceVolume:
		if match := managedDiskPathRE.FindString(sourceResourceID); match == "" {
			sourceResourceID = fmt.Sprintf(managedDiskPath, subscriptionID, resourceGroup, sourceResourceID)
		}
	default:
		return compute.CreationData{
			CreateOption: compute.Empty,
		}, nil
	}

	splits := strings.Split(sourceResourceID, "/")
	if len(splits) > 9 {
		if sourceType == sourceSnapshot {
			return compute.CreationData{}, fmt.Errorf("sourceResourceID(%s) is invalid, correct format: %s", sourceResourceID, diskSnapshotPathRE)
		}
		return compute.CreationData{}, fmt.Errorf("sourceResourceID(%s) is invalid, correct format: %s", sourceResourceID, managedDiskPathRE)
	}
	return compute.CreationData{
		CreateOption:     compute.Copy,
		SourceResourceID: &sourceResourceID,
	}, nil
}

func isInstanceNotFoundError(err error) bool {
	errMsg := strings.ToLower(err.Error())
	return strings.Contains(errMsg, errStatusCode400) && strings.Contains(errMsg, errInvalidParameter) && strings.Contains(errMsg, errTargetInstanceIds)
}
