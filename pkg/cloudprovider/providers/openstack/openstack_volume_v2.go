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

package openstack

import (
	"time"

	"github.com/gophercloud/gophercloud"
	volumes_v2 "github.com/gophercloud/gophercloud/openstack/blockstorage/v2/volumes"

	"github.com/golang/glog"
)

// Volumes implementation for v2
type VolumesV2 struct {
	blockstorage *gophercloud.ServiceClient
	opts         BlockStorageOpts
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
