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

	k8s_volume "k8s.io/kubernetes/pkg/volume"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"
	volumes_v1 "github.com/gophercloud/gophercloud/openstack/blockstorage/v1/volumes"
	volumes_v2 "github.com/gophercloud/gophercloud/openstack/blockstorage/v2/volumes"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/volumeattach"
	"github.com/gophercloud/gophercloud/pagination"

	"github.com/golang/glog"
)

type volumeService interface {
	createVolume(opts VolumeCreateOpts) (string, error)
	getVolume(diskName string) (Volume, error)
	deleteVolume(volumeName string) error
}

// Volumes implementation for v1
type VolumesV1 struct {
	blockstorage *gophercloud.ServiceClient
	opts         BlockStorageOpts
}

// Volumes implementation for v2
type VolumesV2 struct {
	blockstorage *gophercloud.ServiceClient
	opts         BlockStorageOpts
}

type Volume struct {
	// ID of the instance, to which this volume is attached. "" if not attached
	AttachedServerId string
	// Device file path
	AttachedDevice string
	// Unique identifier for the volume.
	ID string
	// Human-readable display name for the volume.
	Name string
	// Current status of the volume.
	Status string
}

type VolumeCreateOpts struct {
	Size         int
	Availability string
	Name         string
	VolumeType   string
	Metadata     map[string]string
}

func (volumes *VolumesV1) createVolume(opts VolumeCreateOpts) (string, error) {

	create_opts := volumes_v1.CreateOpts{
		Name:         opts.Name,
		Size:         opts.Size,
		VolumeType:   opts.VolumeType,
		Availability: opts.Availability,
		Metadata:     opts.Metadata,
	}

	vol, err := volumes_v1.Create(volumes.blockstorage, create_opts).Extract()
	if err != nil {
		return "", err
	}
	return vol.ID, nil
}

func (volumes *VolumesV2) createVolume(opts VolumeCreateOpts) (string, error) {

	create_opts := volumes_v2.CreateOpts{
		Name:             opts.Name,
		Size:             opts.Size,
		VolumeType:       opts.VolumeType,
		AvailabilityZone: opts.Availability,
		Metadata:         opts.Metadata,
	}

	vol, err := volumes_v2.Create(volumes.blockstorage, create_opts).Extract()
	if err != nil {
		return "", err
	}
	return vol.ID, nil
}

func (volumes *VolumesV1) getVolume(diskName string) (Volume, error) {
	var volume_v1 volumes_v1.Volume
	var volume Volume
	err := volumes_v1.List(volumes.blockstorage, nil).EachPage(func(page pagination.Page) (bool, error) {
		vols, err := volumes_v1.ExtractVolumes(page)
		if err != nil {
			glog.Errorf("Failed to extract volumes: %v", err)
			return false, err
		} else {
			for _, v := range vols {
				glog.V(4).Infof("%s %s %v", v.ID, v.Name, v.Attachments)
				if v.Name == diskName || strings.Contains(v.ID, diskName) {
					volume_v1 = v
					return true, nil
				}
			}
		}
		// if it reached here then no disk with the given name was found.
		errmsg := fmt.Sprintf("Unable to find disk: %s", diskName)
		return false, errors.New(errmsg)
	})
	if err != nil {
		glog.Errorf("Error occurred getting volume: %s", diskName)
		return volume, err
	}

	volume.ID = volume_v1.ID
	volume.Name = volume_v1.Name
	volume.Status = volume_v1.Status

	if len(volume_v1.Attachments) > 0 && volume_v1.Attachments[0]["server_id"] != nil {
		volume.AttachedServerId = volume_v1.Attachments[0]["server_id"].(string)
		volume.AttachedDevice = volume_v1.Attachments[0]["device"].(string)
	}

	return volume, nil
}

func (volumes *VolumesV2) getVolume(diskName string) (Volume, error) {
	var volume_v2 volumes_v2.Volume
	var volume Volume
	err := volumes_v2.List(volumes.blockstorage, nil).EachPage(func(page pagination.Page) (bool, error) {
		vols, err := volumes_v2.ExtractVolumes(page)
		if err != nil {
			glog.Errorf("Failed to extract volumes: %v", err)
			return false, err
		} else {
			for _, v := range vols {
				glog.V(4).Infof("%s %s %v", v.ID, v.Name, v.Attachments)
				if v.Name == diskName || strings.Contains(v.ID, diskName) {
					volume_v2 = v
					return true, nil
				}
			}
		}
		// if it reached here then no disk with the given name was found.
		errmsg := fmt.Sprintf("Unable to find disk: %s", diskName)
		return false, errors.New(errmsg)
	})
	if err != nil {
		glog.Errorf("Error occurred getting volume: %s", diskName)
		return volume, err
	}

	volume.ID = volume_v2.ID
	volume.Name = volume_v2.Name
	volume.Status = volume_v2.Status

	if len(volume_v2.Attachments) > 0 {
		volume.AttachedServerId = volume_v2.Attachments[0].ServerID
		volume.AttachedDevice = volume_v2.Attachments[0].Device
	}

	return volume, nil
}

func (volumes *VolumesV1) deleteVolume(volumeName string) error {

	err := volumes_v1.Delete(volumes.blockstorage, volumeName).ExtractErr()
	if err != nil {
		glog.Errorf("Cannot delete volume %s: %v", volumeName, err)
	}
	return err
}

func (volumes *VolumesV2) deleteVolume(volumeName string) error {
	err := volumes_v2.Delete(volumes.blockstorage, volumeName).ExtractErr()
	if err != nil {
		glog.Errorf("Cannot delete volume %s: %v", volumeName, err)
	}
	return err
}

// Attaches given cinder volume to the compute running kubelet
func (os *OpenStack) AttachDisk(instanceID string, diskName string) (string, error) {
	volume, err := os.getVolume(diskName)
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

	if volume.AttachedServerId != "" {
		if instanceID == volume.AttachedServerId {
			glog.V(4).Infof("Disk: %q is already attached to compute: %q", diskName, instanceID)
			return volume.ID, nil
		}
		glog.V(2).Infof("Disk %q is attached to a different compute (%q), detaching", diskName, volume.AttachedServerId)
		err = os.DetachDisk(volume.AttachedServerId, diskName)
		if err != nil {
			return "", err
		}
	}

	// add read only flag here if possible spothanis
	_, err = volumeattach.Create(cClient, instanceID, &volumeattach.CreateOpts{
		VolumeID: volume.ID,
	}).Extract()
	if err != nil {
		glog.Errorf("Failed to attach %s volume to %s compute: %v", diskName, instanceID, err)
		return "", err
	}
	glog.V(2).Infof("Successfully attached %s volume to %s compute", diskName, instanceID)
	return volume.ID, nil
}

// Detaches given cinder volume from the compute running kubelet
func (os *OpenStack) DetachDisk(instanceID string, partialDiskId string) error {
	volume, err := os.getVolume(partialDiskId)
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
	if volume.AttachedServerId != instanceID {
		errMsg := fmt.Sprintf("Disk: %s has no attachments or is not attached to compute: %s", volume.Name, instanceID)
		glog.Errorf(errMsg)
		return errors.New(errMsg)
	} else {
		// This is a blocking call and effects kubelet's performance directly.
		// We should consider kicking it out into a separate routine, if it is bad.
		err = volumeattach.Delete(cClient, instanceID, volume.ID).ExtractErr()
		if err != nil {
			glog.Errorf("Failed to delete volume %s from compute %s attached %v", volume.ID, instanceID, err)
			return err
		}
		glog.V(2).Infof("Successfully detached volume: %s from compute: %s", volume.ID, instanceID)
	}

	return nil
}

// Takes a partial/full disk id or diskname
func (os *OpenStack) getVolume(diskName string) (Volume, error) {

	volumes, err := os.volumeService("")
	if err != nil || volumes == nil {
		glog.Errorf("Unable to initialize cinder client for region: %s", os.region)
		return Volume{}, err
	}

	return volumes.getVolume(diskName)
}

// Create a volume of given size (in GiB)
func (os *OpenStack) CreateVolume(name string, size int, vtype, availability string, tags *map[string]string) (volumeName string, err error) {

	volumes, err := os.volumeService("")
	if err != nil || volumes == nil {
		glog.Errorf("Unable to initialize cinder client for region: %s", os.region)
		return "", err
	}
	opts := VolumeCreateOpts{
		Name:         name,
		Size:         size,
		VolumeType:   vtype,
		Availability: availability,
	}
	if tags != nil {
		opts.Metadata = *tags
	}
	volume_id, err := volumes.createVolume(opts)

	if err != nil {
		glog.Errorf("Failed to create a %d GB volume: %v", size, err)
		return "", err
	}

	glog.Infof("Created volume %v", volume_id)
	return volume_id, nil
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
		return k8s_volume.NewDeletedVolumeInUseError(msg)
	}

	volumes, err := os.volumeService("")
	if err != nil || volumes == nil {
		glog.Errorf("Unable to initialize cinder client for region: %s", os.region)
		return err
	}

	err = volumes.deleteVolume(volumeName)
	if err != nil {
		glog.Errorf("Cannot delete volume %s: %v", volumeName, err)
	}
	return nil

}

// Get device path of attached volume to the compute running kubelet, as known by cinder
func (os *OpenStack) GetAttachmentDiskPath(instanceID string, diskName string) (string, error) {
	// See issue #33128 - Cinder does not always tell you the right device path, as such
	// we must only use this value as a last resort.
	volume, err := os.getVolume(diskName)
	if err != nil {
		return "", err
	}
	if volume.AttachedServerId != "" {
		if instanceID == volume.AttachedServerId {
			// Attachment[0]["device"] points to the device path
			// see http://developer.openstack.org/api-ref-blockstorage-v1.html
			return volume.AttachedDevice, nil
		} else {
			errMsg := fmt.Sprintf("Disk %q is attached to a different compute: %q, should be detached before proceeding", diskName, volume.AttachedServerId)
			glog.Errorf(errMsg)
			return "", errors.New(errMsg)
		}
	}
	return "", fmt.Errorf("volume %s is not attached to %s", diskName, instanceID)
}

// query if a volume is attached to a compute instance
func (os *OpenStack) DiskIsAttached(diskName, instanceID string) (bool, error) {
	volume, err := os.getVolume(diskName)
	if err != nil {
		return false, err
	}

	if instanceID == volume.AttachedServerId {
		return true, nil
	}
	return false, nil
}

// query if a list of volumes are attached to a compute instance
func (os *OpenStack) DisksAreAttached(diskNames []string, instanceID string) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, diskName := range diskNames {
		is_attached, _ := os.DiskIsAttached(diskName, instanceID)
		attached[diskName] = is_attached
	}
	return attached, nil
}

// diskIsUsed returns true a disk is attached to any node.
func (os *OpenStack) diskIsUsed(diskName string) (bool, error) {
	volume, err := os.getVolume(diskName)
	if err != nil {
		return false, err
	}
	if volume.AttachedServerId != "" {
		return true, nil
	}
	return false, nil
}

// query if we should trust the cinder provide deviceName, See issue #33128
func (os *OpenStack) ShouldTrustDevicePath() bool {
	return os.bsOpts.TrustDevicePath
}
