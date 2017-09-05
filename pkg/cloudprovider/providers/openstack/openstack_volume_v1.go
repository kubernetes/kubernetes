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
	volumes_v1 "github.com/gophercloud/gophercloud/openstack/blockstorage/v1/volumes"

	"github.com/golang/glog"
)

// Volumes implementation for v1
type VolumesV1 struct {
	blockstorage *gophercloud.ServiceClient
	opts         BlockStorageOpts
}

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
