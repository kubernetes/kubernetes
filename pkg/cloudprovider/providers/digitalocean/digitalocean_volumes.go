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

package digitalocean

import (
	"errors"
	"fmt"
	"io/ioutil"
	"path"
	"strings"
	"time"

	"github.com/digitalocean/godo"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume"
)

var ErrVolumeNotFound = errors.New("Failed to find volume")
var ErrVolumePendingOperation = errors.New("Volume has pending operation")

const (
	checkSleepDuration = time.Second * 2 // check every 2 seconds
)
const (
	checkTimeout = time.Second * 60 // for max 60 seconds
)

// Create a volume of given size (in GiB)
func (do *DigitalOcean) CreateVolume(region string, name string, description string, sizeGigaBytes int64) (volumeName string, err error) {
	volumeCreateRequest := godo.VolumeCreateRequest{
		Region:        region,
		Name:          name,
		Description:   description,
		SizeGigaBytes: sizeGigaBytes,
	}
	vol, _, err := do.provider.Storage.CreateVolume(&volumeCreateRequest)
	if err != nil {
		glog.Errorf("Failed to create a %d GB volume: %v", volumeCreateRequest.SizeGigaBytes, err)
		return "", err
	}
	glog.Infof("Created volume %v", vol.ID)
	return vol.ID, err
}

// Delete a volume
func (do *DigitalOcean) DeleteVolume(volumeID string) error {
	used, err := do.volumeIsUsed(volumeID)
	if err != nil {
		return err
	}
	if used {
		msg := fmt.Sprintf("Cannot delete the volume %s, it's still attached to a node", volumeID)
		return volume.NewDeletedVolumeInUseError(msg)
	}

	_, err = do.provider.Storage.DeleteVolume(volumeID)
	if err != nil {
		glog.Errorf("Cannot delete volume %s: %v", volumeID, err)
	}
	return err
}

// volumeIsUsed returns true if a volume is attached to a node.
func (do *DigitalOcean) volumeIsUsed(volumeID string) (bool, error) {
	volume, _, err := do.provider.Storage.GetVolume(volumeID)
	if err != nil {
		return false, err
	}
	if len(volume.DropletIDs) > 0 {
		return true, nil
	}
	return false, nil
}

// Attaches given DigitalOcean volume
func (do *DigitalOcean) AttachVolume(instanceID int, volumeID string) (string, error) {
	// volumeID = kubernetes volume ID
	// volume.ID = DigitalOcean volume ID

	volume, err := do.getVolume(volumeID)
	if err != nil {
		glog.Errorf("Failed to get DigitalOcean volume for volume: %s", volumeID)
		return "", err
	}

	for _, dropletID := range volume.DropletIDs {
		if instanceID == dropletID {
			glog.V(2).Infof("Volume %s is already attached to %s compute, not reattaching", volumeID, instanceID)
			return volume.ID, nil
		}
	}
	if len(volume.DropletIDs) > 0 {
		// There's still a volume attached, check whether detach is still in progress
		err = wait.Poll(checkSleepDuration, checkTimeout, func() (bool, error) {
			listOptions := &godo.ListOptions{
				Page:    1,
				PerPage: 1,
			}
			actions, _, err := do.provider.StorageActions.List(volume.ID, listOptions)
			if err != nil {
				glog.V(2).Infof("Failed to get pending led to attach volumeactions for volume %s (%s)", volumeID, volume.ID)
			}
			for _, action := range actions {
				if action.Status == "completed" {
					return true, nil
				} else {
					glog.V(2).Infof("Volume %s (%s) still has an operation pending (action: %s, status: %s)", volumeID, volume.ID, action.Type, action.Status)
					return false, ErrVolumePendingOperation
				}
			}
			// no action pending
			return true, nil
		})
		if err != nil {
			glog.V(2).Infof("Volume %s (%s) still has a pending operation, will try to attach anyway", volumeID, volume.ID)
		}
	}

	_, _, err = do.provider.StorageActions.Attach(volume.ID, instanceID)
	if err != nil {
		glog.Errorf("Failed to attach %s (%s) volume to %s compute", volumeID, volume.ID, instanceID)
		return "", err
	}
	glog.V(2).Infof("Successfully attached %s volume to %s compute", volumeID, instanceID)
	return volume.ID, nil
}

// Detaches given DigitalOcean volume from the compute running kubelet
func (do *DigitalOcean) DetachVolume(instanceID int, volumeID string) error {
	volume, err := do.getVolume(volumeID)
	if err != nil {
		glog.Errorf("Failed to get DigitalOcean volume for volume: %s", volumeID)
		return err
	}
	_, _, err = do.provider.StorageActions.DetachByDropletID(volume.ID, instanceID)
	if err != nil {
		glog.Errorf("Failed to detach %s (%s) volume", volumeID, volume.ID)
		return err
	}
	glog.V(2).Infof("Successfully detached %s (%s) volume", volumeID, volume.ID)
	return nil
}

func (do *DigitalOcean) getVolume(volumeID string) (*godo.Volume, error) {
	listOptions := &godo.ListOptions{
		Page:    1,
		PerPage: 200,
	}
	volumes, _, err := do.provider.Storage.ListVolumes(listOptions)
	if err != nil {
		glog.Errorf("Error occurred getting volume: %s", volumeID)
		return nil, err
	}
	for _, volume := range volumes {
		if volume.Name == volumeID || volume.ID == volumeID {
			return &volume, nil
		}
	}
	glog.Errorf("Volume not found: %s", volumeID)
	return nil, ErrVolumeNotFound
}

// GetDevicePath returns the path of an attached block storage volume, specified by its id.
func (do *DigitalOcean) GetDevicePath(volumeId string) string {
	files, _ := ioutil.ReadDir("/dev/disk/by-id/")
	for _, f := range files {
		if strings.Contains(f.Name(), "scsi-0DO_Volume_") {
			devid_prefix := f.Name()[len("scsi-0DO_Volume_"):len(f.Name())]
			if strings.Contains(volumeId, devid_prefix) {
				glog.V(4).Infof("Found disk attached as %q; full devicepath: %s\n", f.Name(), path.Join("/dev/disk/by-id/", f.Name()))
				return path.Join("/dev/disk/by-id/", f.Name())
			}
		}
	}
	glog.Warningf("Failed to find device for the diskid: %q\n", volumeId)
	return ""
}

// Get device path of attached volume to the compute running kubelet
func (do *DigitalOcean) GetAttachmentVolumePath(instanceID int, volumeID string) (string, error) {
	volume, err := do.getVolume(volumeID)
	if err != nil {
		return "", err
	}
	return "/dev/disk/by-id/scsi-0DO_Volume_" + volume.Name, nil
}

// query if a volume is attached to a compute instance
func (do *DigitalOcean) VolumeIsAttached(volumeID string, instanceID int) (bool, error) {
	volume, err := do.getVolume(volumeID)
	if err != nil {
		return false, err
	}
	if len(volume.DropletIDs) == 0 {
		return false, nil
	}
	attached := false
	for _, i := range volume.DropletIDs {
		if i == instanceID {
			attached = true
		}
	}
	return attached, nil
}

// query if a list of volumes are attached to a compute instance
func (do *DigitalOcean) VolumesAreAttached(volumeIDs []string, instanceID int) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, volumeID := range volumeIDs {
		attached[volumeID] = false
	}
	for _, volumeID := range volumeIDs {
		volume, err := do.getVolume(volumeID)
		if err != nil {
			continue
		}
		for _, i := range volume.DropletIDs {
			if i == instanceID {
				attached[volumeID] = true
			}
		}
	}
	return attached, nil
}
