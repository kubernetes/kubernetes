/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package qingcloud

// See https://docs.qingcloud.com/api/volume/index.html

import (
	"fmt"
	"strings"

	"github.com/magicshui/qingcloud-go"
	"github.com/magicshui/qingcloud-go/volume"
	"k8s.io/kubernetes/pkg/types"
)

// DefaultMaxQingCloudVolumes is the limit for volumes attached to an instance.
// TODO: No clear description from qingcloud document
//const DefaultMaxQingCloudVolumes = 6

// VolumeOptions specifies capacity and type for a volume.
// See https://docs.qingcloud.com/api/volume/create_volumes.html
type VolumeOptions struct {
	CapacityGB int // minimum 10GiB, maximum 500GiB, must be a multiple of 10x
	VolumeType int // only can be 0, 1, 2, 3
}

// Volumes is an interface for managing cloud-provisioned volumes
type Volumes interface {
	// Attach the disk to the specified instance
	// Returns the device (e.g. /dev/sdb) where we attached the volume
	// It checks if volume is already attached to node and succeeds in that case.
	AttachVolume(volumeID string, nodeName types.NodeName) (string, error)

	// Detach the disk from the specified instance
	DetachVolume(volumeID string, nodeName types.NodeName) error

	// Create a volume with the specified options
	CreateVolume(volumeOptions *VolumeOptions) (volumeID string, err error)

	// Delete the specified volume
	// Returns true if the volume was deleted
	// If the was not found, returns (false, nil)
	DeleteVolume(volumeID string) (bool, error)

	// Check if the volume is already attached to the instance
	VolumeIsAttached(volumeID string, nodeName types.NodeName) (bool, error)
}

// AttachVolumes implements Volumes.AttachVolume
func (qc *Qingcloud) AttachVolume(volumeID string, nodeName types.NodeName) (string, error) {
	attached, err := qc.VolumeIsAttached(volumeID, nodeName)
	if err != nil {
		return "", err
	}

	volumesN := qingcloud.NumberedString{}
	volumesN.Add(volumeID)

	if !attached {
		instance := qingcloud.String{}
		instance.Set(nodeNameToInstanceId(nodeName))
		_, err := qc.volumeClient.AttachVolumes(volume.AttachVolumesRequest{
			VolumesN: volumesN,
			Instance: instance,
		})
		if err != nil {
			return "", err
		}
	}

	resp, err := qc.volumeClient.DescribeVolumes(volume.DescribeVolumesRequest{
		VolumesN: volumesN,
	})
	if err != nil {
		return "", err
	}
	if len(resp.VolumeSet) == 0 {
		return "", fmt.Errorf("volume '%v' miss after attach it", volumeID)
	}

	dev := resp.VolumeSet[0].Instance.Device
	if dev == "" {
		return "", fmt.Errorf("the device of volume '%v' is empty", volumeID)
	}

	return dev, nil
}

// DetachVolumes implements Volumes.DetachVolume
func (qc *Qingcloud) DetachVolume(volumeID string, nodeName types.NodeName) error {
	volumesN := qingcloud.NumberedString{}
	volumesN.Add(volumeID)
	instance := qingcloud.String{}
	instance.Set(nodeNameToInstanceId(nodeName))
	_, err := qc.volumeClient.DetachVolumes(volume.DetachVolumesRequest{
		VolumesN: volumesN,
		Instance: instance,
	})

	return err
}

// CreateVolumes implements Volumes.CreateVolume
func (qc *Qingcloud) CreateVolume(volumeOptions *VolumeOptions) (string, error) {
	vsize := qingcloud.Integer{}
	vsize.Set(volumeOptions.CapacityGB)
	vtype := qingcloud.Integer{}
	vtype.Set(volumeOptions.VolumeType)
	resp, err := qc.volumeClient.CreateVolumes(volume.CreateVolumesRequest{
		Size:       vsize,
		VolumeType: vtype,
	})
	if err != nil {
		return "", err
	}

	return resp.Volumes[0], nil
}

// DeleteVolumes implements Volumes.DeleteVolume
func (qc *Qingcloud) DeleteVolume(volumeID string) (bool, error) {
	volumesN := qingcloud.NumberedString{}
	volumesN.Add(volumeID)
	_, err := qc.volumeClient.DeleteVolumes(volume.DeleteVolumesRequest{
		VolumesN: volumesN,
	})
	if err != nil {
		if strings.Index(err.Error(), "already been deleted") >= 0 {
			return false, nil
		}
		return false, err
	}

	return true, nil
}

// VolumeIsAttached implements Volumes.VolumeIsAttached
func (qc *Qingcloud) VolumeIsAttached(volumeID string, nodeName types.NodeName) (bool, error) {
	volumesN := qingcloud.NumberedString{}
	volumesN.Add(volumeID)
	resp, err := qc.volumeClient.DescribeVolumes(volume.DescribeVolumesRequest{
		VolumesN: volumesN,
	})
	if err != nil {
		return false, err
	}
	if len(resp.VolumeSet) == 0 {
		return false, nil
	}

	return resp.VolumeSet[0].Instance.InstanceID == nodeNameToInstanceId(nodeName), nil
}
