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
	"time"

	k8s_volume "k8s.io/kubernetes/pkg/volume"

	"github.com/gophercloud/gophercloud"
	volumes_v1 "github.com/gophercloud/gophercloud/openstack/blockstorage/v1/volumes"
	volumes_v2 "github.com/gophercloud/gophercloud/openstack/blockstorage/v2/volumes"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/volumeattach"
	"github.com/prometheus/client_golang/prometheus"

	"github.com/golang/glog"
)

type volumeService interface {
	createVolume(opts VolumeCreateOpts) (string, string, error)
	getVolume(volumeID string) (Volume, error)
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

const (
	VolumeAvailableStatus = "available"
	VolumeInUseStatus     = "in-use"
	VolumeDeletedStatus   = "deleted"
	VolumeErrorStatus     = "error"
)

func (volumes *VolumesV1) createVolume(opts VolumeCreateOpts) (string, string, error) {
	startTime := time.Now()

	create_opts := volumes_v1.CreateOpts{
		Name:             opts.Name,
		Size:             opts.Size,
		VolumeType:       opts.VolumeType,
		AvailabilityZone: opts.Availability,
		Metadata:         opts.Metadata,
	}

	vol, err := volumes_v1.Create(volumes.blockstorage, create_opts).Extract()
	timeTaken := time.Since(startTime).Seconds()
	recordOpenstackOperationMetric("create_v1_volume", timeTaken, err)
	if err != nil {
		return "", "", err
	}
	return vol.ID, vol.AvailabilityZone, nil
}

func (volumes *VolumesV2) createVolume(opts VolumeCreateOpts) (string, string, error) {
	startTime := time.Now()

	create_opts := volumes_v2.CreateOpts{
		Name:             opts.Name,
		Size:             opts.Size,
		VolumeType:       opts.VolumeType,
		AvailabilityZone: opts.Availability,
		Metadata:         opts.Metadata,
	}

	vol, err := volumes_v2.Create(volumes.blockstorage, create_opts).Extract()
	timeTaken := time.Since(startTime).Seconds()
	recordOpenstackOperationMetric("create_v2_volume", timeTaken, err)
	if err != nil {
		return "", "", err
	}
	return vol.ID, vol.AvailabilityZone, nil
}

func (volumes *VolumesV1) getVolume(volumeID string) (Volume, error) {
	startTime := time.Now()
	volumeV1, err := volumes_v1.Get(volumes.blockstorage, volumeID).Extract()
	timeTaken := time.Since(startTime).Seconds()
	recordOpenstackOperationMetric("get_v1_volume", timeTaken, err)
	if err != nil {
		glog.Errorf("Error occurred getting volume by ID: %s", volumeID)
		return Volume{}, err
	}

	volume := Volume{
		ID:     volumeV1.ID,
		Name:   volumeV1.Name,
		Status: volumeV1.Status,
	}

	if len(volumeV1.Attachments) > 0 && volumeV1.Attachments[0]["server_id"] != nil {
		volume.AttachedServerId = volumeV1.Attachments[0]["server_id"].(string)
		volume.AttachedDevice = volumeV1.Attachments[0]["device"].(string)
	}

	return volume, nil
}

func (volumes *VolumesV2) getVolume(volumeID string) (Volume, error) {
	startTime := time.Now()
	volumeV2, err := volumes_v2.Get(volumes.blockstorage, volumeID).Extract()
	timeTaken := time.Since(startTime).Seconds()
	recordOpenstackOperationMetric("get_v2_volume", timeTaken, err)
	if err != nil {
		glog.Errorf("Error occurred getting volume by ID: %s", volumeID)
		return Volume{}, err
	}

	volume := Volume{
		ID:     volumeV2.ID,
		Name:   volumeV2.Name,
		Status: volumeV2.Status,
	}

	if len(volumeV2.Attachments) > 0 {
		volume.AttachedServerId = volumeV2.Attachments[0].ServerID
		volume.AttachedDevice = volumeV2.Attachments[0].Device
	}

	return volume, nil
}

func (volumes *VolumesV1) deleteVolume(volumeID string) error {
	startTime := time.Now()
	err := volumes_v1.Delete(volumes.blockstorage, volumeID).ExtractErr()
	timeTaken := time.Since(startTime).Seconds()
	recordOpenstackOperationMetric("delete_v1_volume", timeTaken, err)
	if err != nil {
		glog.Errorf("Cannot delete volume %s: %v", volumeID, err)
	}

	return err
}

func (volumes *VolumesV2) deleteVolume(volumeID string) error {
	startTime := time.Now()
	err := volumes_v2.Delete(volumes.blockstorage, volumeID).ExtractErr()
	timeTaken := time.Since(startTime).Seconds()
	recordOpenstackOperationMetric("delete_v2_volume", timeTaken, err)
	if err != nil {
		glog.Errorf("Cannot delete volume %s: %v", volumeID, err)
	}

	return err
}

func (os *OpenStack) OperationPending(diskName string) (bool, string, error) {
	volume, err := os.getVolume(diskName)
	if err != nil {
		return false, "", err
	}
	volumeStatus := volume.Status
	if volumeStatus == VolumeErrorStatus {
		glog.Errorf("status of volume %s is %s", diskName, volumeStatus)
		return false, volumeStatus, nil
	}
	if volumeStatus == VolumeAvailableStatus || volumeStatus == VolumeInUseStatus || volumeStatus == VolumeDeletedStatus {
		return false, volume.Status, nil
	}
	return true, volumeStatus, nil
}

// Attaches given cinder volume to the compute running kubelet
func (os *OpenStack) AttachDisk(instanceID, volumeID string) (string, error) {
	volume, err := os.getVolume(volumeID)
	if err != nil {
		return "", err
	}

	cClient, err := os.NewComputeV2()
	if err != nil {
		return "", err
	}

	if volume.AttachedServerId != "" {
		if instanceID == volume.AttachedServerId {
			glog.V(4).Infof("Disk %s is already attached to instance %s", volumeID, instanceID)
			return volume.ID, nil
		}
		errmsg := fmt.Sprintf("Disk %s is attached to a different instance (%s)", volumeID, volume.AttachedServerId)
		glog.V(2).Infof(errmsg)
		return "", errors.New(errmsg)
	}

	startTime := time.Now()
	// add read only flag here if possible spothanis
	_, err = volumeattach.Create(cClient, instanceID, &volumeattach.CreateOpts{
		VolumeID: volume.ID,
	}).Extract()
	timeTaken := time.Since(startTime).Seconds()
	recordOpenstackOperationMetric("attach_disk", timeTaken, err)
	if err != nil {
		glog.Errorf("Failed to attach %s volume to %s compute: %v", volumeID, instanceID, err)
		return "", err
	}
	glog.V(2).Infof("Successfully attached %s volume to %s compute", volumeID, instanceID)
	return volume.ID, nil
}

// DetachDisk detaches given cinder volume from the compute running kubelet
func (os *OpenStack) DetachDisk(instanceID, volumeID string) error {
	volume, err := os.getVolume(volumeID)
	if err != nil {
		return err
	}
	if volume.Status == VolumeAvailableStatus {
		// "available" is fine since that means the volume is detached from instance already.
		glog.V(2).Infof("volume: %s has been detached from compute: %s ", volume.ID, instanceID)
		return nil
	}

	if volume.Status != VolumeInUseStatus {
		errmsg := fmt.Sprintf("can not detach volume %s, its status is %s.", volume.Name, volume.Status)
		glog.Errorf(errmsg)
		return errors.New(errmsg)
	}
	cClient, err := os.NewComputeV2()
	if err != nil {
		return err
	}
	if volume.AttachedServerId != instanceID {
		errMsg := fmt.Sprintf("Disk: %s has no attachments or is not attached to compute: %s", volume.Name, instanceID)
		glog.Errorf(errMsg)
		return errors.New(errMsg)
	} else {
		startTime := time.Now()
		// This is a blocking call and effects kubelet's performance directly.
		// We should consider kicking it out into a separate routine, if it is bad.
		err = volumeattach.Delete(cClient, instanceID, volume.ID).ExtractErr()
		timeTaken := time.Since(startTime).Seconds()
		recordOpenstackOperationMetric("detach_disk", timeTaken, err)
		if err != nil {
			glog.Errorf("Failed to delete volume %s from compute %s attached %v", volume.ID, instanceID, err)
			return err
		}
		glog.V(2).Infof("Successfully detached volume: %s from compute: %s", volume.ID, instanceID)
	}

	return nil
}

// Retrieves Volume by its ID.
func (os *OpenStack) getVolume(volumeID string) (Volume, error) {
	volumes, err := os.volumeService("")
	if err != nil || volumes == nil {
		glog.Errorf("Unable to initialize cinder client for region: %s", os.region)
		return Volume{}, err
	}
	return volumes.getVolume(volumeID)
}

// Create a volume of given size (in GiB)
func (os *OpenStack) CreateVolume(name string, size int, vtype, availability string, tags *map[string]string) (string, string, error) {
	volumes, err := os.volumeService("")
	if err != nil || volumes == nil {
		glog.Errorf("Unable to initialize cinder client for region: %s", os.region)
		return "", "", err
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

	volumeID, volumeAZ, err := volumes.createVolume(opts)

	if err != nil {
		glog.Errorf("Failed to create a %d GB volume: %v", size, err)
		return "", "", err
	}

	glog.Infof("Created volume %v in Availability Zone: %v", volumeID, volumeAZ)
	return volumeID, volumeAZ, nil
}

// GetDevicePath returns the path of an attached block storage volume, specified by its id.
func (os *OpenStack) GetDevicePath(volumeID string) string {
	// Build a list of candidate device paths
	candidateDeviceNodes := []string{
		// KVM
		fmt.Sprintf("virtio-%s", volumeID[:20]),
		// KVM virtio-scsi
		fmt.Sprintf("scsi-0QEMU_QEMU_HARDDISK_%s", volumeID[:20]),
		// ESXi
		fmt.Sprintf("wwn-0x%s", strings.Replace(volumeID, "-", "", -1)),
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

	glog.Warningf("Failed to find device for the volumeID: %q\n", volumeID)
	return ""
}

func (os *OpenStack) DeleteVolume(volumeID string) error {
	used, err := os.diskIsUsed(volumeID)
	if err != nil {
		return err
	}
	if used {
		msg := fmt.Sprintf("Cannot delete the volume %q, it's still attached to a node", volumeID)
		return k8s_volume.NewDeletedVolumeInUseError(msg)
	}

	volumes, err := os.volumeService("")
	if err != nil || volumes == nil {
		glog.Errorf("Unable to initialize cinder client for region: %s", os.region)
		return err
	}

	err = volumes.deleteVolume(volumeID)
	if err != nil {
		glog.Errorf("Cannot delete volume %s: %v", volumeID, err)
	}
	return nil

}

// Get device path of attached volume to the compute running kubelet, as known by cinder
func (os *OpenStack) GetAttachmentDiskPath(instanceID, volumeID string) (string, error) {
	// See issue #33128 - Cinder does not always tell you the right device path, as such
	// we must only use this value as a last resort.
	volume, err := os.getVolume(volumeID)
	if err != nil {
		return "", err
	}
	if volume.Status != VolumeInUseStatus {
		errmsg := fmt.Sprintf("can not get device path of volume %s, its status is %s.", volume.Name, volume.Status)
		glog.Errorf(errmsg)
		return "", errors.New(errmsg)
	}
	if volume.AttachedServerId != "" {
		if instanceID == volume.AttachedServerId {
			// Attachment[0]["device"] points to the device path
			// see http://developer.openstack.org/api-ref-blockstorage-v1.html
			return volume.AttachedDevice, nil
		} else {
			errMsg := fmt.Sprintf("Disk %q is attached to a different compute: %q, should be detached before proceeding", volumeID, volume.AttachedServerId)
			glog.Errorf(errMsg)
			return "", errors.New(errMsg)
		}
	}
	return "", fmt.Errorf("volume %s has no ServerId.", volumeID)
}

// query if a volume is attached to a compute instance
func (os *OpenStack) DiskIsAttached(instanceID, volumeID string) (bool, error) {
	volume, err := os.getVolume(volumeID)
	if err != nil {
		return false, err
	}

	return instanceID == volume.AttachedServerId, nil
}

// query if a list of volumes are attached to a compute instance
func (os *OpenStack) DisksAreAttached(instanceID string, volumeIDs []string) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, volumeID := range volumeIDs {
		isAttached, _ := os.DiskIsAttached(instanceID, volumeID)
		attached[volumeID] = isAttached
	}
	return attached, nil
}

// diskIsUsed returns true a disk is attached to any node.
func (os *OpenStack) diskIsUsed(volumeID string) (bool, error) {
	volume, err := os.getVolume(volumeID)
	if err != nil {
		return false, err
	}
	return volume.AttachedServerId != "", nil
}

// query if we should trust the cinder provide deviceName, See issue #33128
func (os *OpenStack) ShouldTrustDevicePath() bool {
	return os.bsOpts.TrustDevicePath
}

// recordOpenstackOperationMetric records openstack operation metrics
func recordOpenstackOperationMetric(operation string, timeTaken float64, err error) {
	if err != nil {
		OpenstackApiRequestErrors.With(prometheus.Labels{"request": operation}).Inc()
	} else {
		OpenstackOperationsLatency.With(prometheus.Labels{"request": operation}).Observe(timeTaken)
	}
}
