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

package openstack

import (
	"errors"
	"fmt"
	"io/ioutil"
	"path"
	"strings"

	"k8s.io/kubernetes/pkg/volume"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack"
	"github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/volumeattach"
	"github.com/rackspace/gophercloud/pagination"

	"github.com/golang/glog"
)

// Attaches given cinder volume to the compute running kubelet
func (os *OpenStack) AttachDisk(instanceID string, diskName string) (string, error) {
	disk, err := os.getVolume(diskName)
	if err != nil {
		return "", err
	}
	cClient, err := openstack.NewComputeV2(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil || cClient == nil {
		glog.Errorf("Unable to initialize nova client for region: %s", os.region)
		return "", err
	}

	if len(disk.Attachments) > 0 && disk.Attachments[0]["server_id"] != nil {
		if instanceID == disk.Attachments[0]["server_id"] {
			glog.V(4).Infof("Disk: %q is already attached to compute: %q", diskName, instanceID)
			return disk.ID, nil
		}

		glog.V(2).Infof("Disk %q is attached to a different compute (%q), detaching", diskName, disk.Attachments[0]["server_id"])
		err = os.DetachDisk(fmt.Sprintf("%s", disk.Attachments[0]["server_id"]), diskName)
		if err != nil {
			return "", err
		}
	}

	// add read only flag here if possible spothanis
	_, err = volumeattach.Create(cClient, instanceID, &volumeattach.CreateOpts{
		VolumeID: disk.ID,
	}).Extract()
	if err != nil {
		glog.Errorf("Failed to attach %s volume to %s compute: %v", diskName, instanceID, err)
		return "", err
	}
	glog.V(2).Infof("Successfully attached %s volume to %s compute", diskName, instanceID)
	return disk.ID, nil
}

// Detaches given cinder volume from the compute running kubelet
func (os *OpenStack) DetachDisk(instanceID string, partialDiskId string) error {
	disk, err := os.getVolume(partialDiskId)
	if err != nil {
		return err
	}
	cClient, err := openstack.NewComputeV2(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil || cClient == nil {
		glog.Errorf("Unable to initialize nova client for region: %s", os.region)
		return err
	}
	if len(disk.Attachments) > 0 && disk.Attachments[0]["server_id"] != nil && instanceID == disk.Attachments[0]["server_id"] {
		// This is a blocking call and effects kubelet's performance directly.
		// We should consider kicking it out into a separate routine, if it is bad.
		err = volumeattach.Delete(cClient, instanceID, disk.ID).ExtractErr()
		if err != nil {
			glog.Errorf("Failed to delete volume %s from compute %s attached %v", disk.ID, instanceID, err)
			return err
		}
		glog.V(2).Infof("Successfully detached volume: %s from compute: %s", disk.ID, instanceID)
	} else {
		errMsg := fmt.Sprintf("Disk: %s has no attachments or is not attached to compute: %s", disk.Name, instanceID)
		glog.Errorf(errMsg)
		return errors.New(errMsg)
	}
	return nil
}

// Takes a partial/full disk id or diskname
func (os *OpenStack) getVolume(diskName string) (volumes.Volume, error) {
	sClient, err := openstack.NewBlockStorageV1(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})

	var volume volumes.Volume
	if err != nil || sClient == nil {
		glog.Errorf("Unable to initialize cinder client for region: %s", os.region)
		return volume, err
	}

	err = volumes.List(sClient, nil).EachPage(func(page pagination.Page) (bool, error) {
		vols, err := volumes.ExtractVolumes(page)
		if err != nil {
			glog.Errorf("Failed to extract volumes: %v", err)
			return false, err
		} else {
			for _, v := range vols {
				glog.V(4).Infof("%s %s %v", v.ID, v.Name, v.Attachments)
				if v.Name == diskName || strings.Contains(v.ID, diskName) {
					volume = v
					return true, nil
				}
			}
		}
		// if it reached here then no disk with the given name was found.
		errmsg := fmt.Sprintf("Unable to find disk: %s in region %s", diskName, os.region)
		return false, errors.New(errmsg)
	})
	if err != nil {
		glog.Errorf("Error occurred getting volume: %s", diskName)
		return volume, err
	}
	return volume, err
}

// Create a volume of given size (in GiB)
func (os *OpenStack) CreateVolume(name string, size int, vtype, availability string, tags *map[string]string) (volumeName string, err error) {

	sClient, err := openstack.NewBlockStorageV1(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})

	if err != nil || sClient == nil {
		glog.Errorf("Unable to initialize cinder client for region: %s", os.region)
		return "", err
	}

	opts := volumes.CreateOpts{
		Name:         name,
		Size:         size,
		VolumeType:   vtype,
		Availability: availability,
	}
	if tags != nil {
		opts.Metadata = *tags
	}
	vol, err := volumes.Create(sClient, opts).Extract()
	if err != nil {
		glog.Errorf("Failed to create a %d GB volume: %v", size, err)
		return "", err
	}
	glog.Infof("Created volume %v", vol.ID)
	return vol.ID, err
}

// GetDevicePath returns the path of an attached block storage volume, specified by its id.
func (os *OpenStack) GetDevicePath(diskId string) string {
	// Build a list of candidate device paths
	candidateDeviceNodes := []string{
		// KVM
		fmt.Sprintf("virtio-%s", diskId[:20]),
		// ESXi
		fmt.Sprintf("wwn-0x%s", strings.Replace(diskId, "-", "", -1)),
	}

	files, _ := ioutil.ReadDir("/dev/disk/by-id/")

	for _, f := range files {
		for _, c := range candidateDeviceNodes {
			if c == f.Name() {
				glog.V(4).Infof("Found disk attached as %q; full devicepath: %s\n", f.Name(), path.Join("/dev/disk/by-id/", f.Name()))
				return path.Join("/dev/disk/by-id/", f.Name())
			}
		}
	}

	glog.Warningf("Failed to find device for the diskid: %q\n", diskId)
	return ""
}

func (os *OpenStack) DeleteVolume(volumeName string) error {
	used, err := os.diskIsUsed(volumeName)
	if err != nil {
		return err
	}
	if used {
		msg := fmt.Sprintf("Cannot delete the volume %q, it's still attached to a node", volumeName)
		return volume.NewDeletedVolumeInUseError(msg)
	}

	sClient, err := openstack.NewBlockStorageV1(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})

	if err != nil || sClient == nil {
		glog.Errorf("Unable to initialize cinder client for region: %s", os.region)
		return err
	}
	err = volumes.Delete(sClient, volumeName).ExtractErr()
	if err != nil {
		glog.Errorf("Cannot delete volume %s: %v", volumeName, err)
	}
	return err
}

// Get device path of attached volume to the compute running kubelet, as known by cinder
func (os *OpenStack) GetAttachmentDiskPath(instanceID string, diskName string) (string, error) {
	// See issue #33128 - Cinder does not always tell you the right device path, as such
	// we must only use this value as a last resort.
	disk, err := os.getVolume(diskName)
	if err != nil {
		return "", err
	}
	if len(disk.Attachments) > 0 && disk.Attachments[0]["server_id"] != nil {
		if instanceID == disk.Attachments[0]["server_id"] {
			// Attachment[0]["device"] points to the device path
			// see http://developer.openstack.org/api-ref-blockstorage-v1.html
			return disk.Attachments[0]["device"].(string), nil
		} else {
			errMsg := fmt.Sprintf("Disk %q is attached to a different compute: %q, should be detached before proceeding", diskName, disk.Attachments[0]["server_id"])
			glog.Errorf(errMsg)
			return "", errors.New(errMsg)
		}
	}
	return "", fmt.Errorf("volume %s is not attached to %s", diskName, instanceID)
}

// query if a volume is attached to a compute instance
func (os *OpenStack) DiskIsAttached(diskName, instanceID string) (bool, error) {
	disk, err := os.getVolume(diskName)
	if err != nil {
		return false, err
	}
	if len(disk.Attachments) > 0 && disk.Attachments[0]["server_id"] != nil && instanceID == disk.Attachments[0]["server_id"] {
		return true, nil
	}
	return false, nil
}

// query if a list of volumes are attached to a compute instance
func (os *OpenStack) DisksAreAttached(diskNames []string, instanceID string) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, diskName := range diskNames {
		attached[diskName] = false
	}
	for _, diskName := range diskNames {
		disk, err := os.getVolume(diskName)
		if err != nil {
			continue
		}
		if len(disk.Attachments) > 0 && disk.Attachments[0]["server_id"] != nil && instanceID == disk.Attachments[0]["server_id"] {
			attached[diskName] = true
		}
	}
	return attached, nil
}

// diskIsUsed returns true a disk is attached to any node.
func (os *OpenStack) diskIsUsed(diskName string) (bool, error) {
	disk, err := os.getVolume(diskName)
	if err != nil {
		return false, err
	}
	if len(disk.Attachments) > 0 {
		return true, nil
	}
	return false, nil
}

// query if we should trust the cinder provide deviceName, See issue #33128
func (os *OpenStack) ShouldTrustDevicePath() bool {
	return os.bsOpts.TrustDevicePath
}
