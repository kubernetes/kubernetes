/*
Copyright 2014 The Kubernetes Authors.

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

package qingcloud_volume

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/qingcloud"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	checkSleepDuration = time.Second
)

type QingDiskUtil struct{}

func (util *QingDiskUtil) DeleteVolume(d *qingcloudVolumeDeleter) error {
	var qcVolume qingcloud.Volumes
	var err error
	if qcVolume, err = getCloudProvider(d.qingcloudVolume.plugin.host.GetCloudProvider()); err != nil {
		return err
	}

	deleted, err := qcVolume.DeleteVolume(d.volumeID)
	if err != nil {
		glog.V(2).Infof("Error deleting qingcloud volume %s: %v", d.volumeID, err)
		return err
	}
	if deleted {
		glog.V(2).Infof("Successfully deleted qingcloud volume %s", d.volumeID)
	} else {
		glog.V(2).Infof("Successfully deleted qingcloud volume %s (actually already deleted)", d.volumeID)
	}
	return nil
}

// CreateVolume creates a qingcloud volume.
// Returns: volumeID, volumeSizeGB, error
func (util *QingDiskUtil) CreateVolume(c *qingcloudVolumeProvisioner) (string, int, error) {
	var qcVolume qingcloud.Volumes
	var err error
	if qcVolume, err = getCloudProvider(c.qingcloudVolume.plugin.host.GetCloudProvider()); err != nil {
		return "", 0, err
	}

	capacity := c.options.PVC.Spec.Resources.Requests[api.ResourceName(api.ResourceStorage)]
	requestBytes := capacity.Value()
	// qingcloud works with gigabytes, convert to GiB with rounding up
	requestGB := int(volume.RoundUpSize(requestBytes, 1024*1024*1024))
	// minimum 10GiB, maximum 500GiB
	if requestGB < 10 {
		requestGB = 10
	} else if requestGB > 500 {
		return "", 0, fmt.Errorf("Can't request volume bigger than 500GiB")
	}
	// must be a multiple of 10x
	requestGB += 10 - requestGB%10
	volumeOptions := &qingcloud.VolumeOptions{
		CapacityGB: requestGB,
	}
	// Apply Parameters (case-insensitive). We leave validation of
	// the values to the cloud provider.
	for k, v := range c.options.Parameters {
		switch strings.ToLower(k) {
		case "type":
			if v != "0" && v != "1" && v != "2" && v != "3" {
				return "", 0, fmt.Errorf("invalid option '%q' for volume plugin %s, it only can be 0, 1, 2, 3",
					k, c.plugin.GetPluginName())
			}
			volumeOptions.VolumeType, _ = strconv.Atoi(v)
		default:
			return "", 0, fmt.Errorf("invalid option '%q' for volume plugin %s", k, c.plugin.GetPluginName())
		}
	}

	// TODO: implement PVC.Selector parsing
	if c.options.PVC.Spec.Selector != nil {
		return "", 0, fmt.Errorf("claim.Spec.Selector is not supported for dynamic provisioning on qingcloud")
	}

	volumeID, err := qcVolume.CreateVolume(volumeOptions)
	if err != nil {
		glog.V(2).Infof("Error creating qingcloud volume: %v", err)
		return "", 0, err
	}
	glog.V(2).Infof("Successfully created qingcloud volume %s", volumeID)

	return volumeID, int(requestGB), nil
}

// Return cloud provider
func getCloudProvider(cloudProvider cloudprovider.Interface) (*qingcloud.Qingcloud, error) {
	qingCloudProvider, ok := cloudProvider.(*qingcloud.Qingcloud)
	if !ok || qingCloudProvider == nil {
		return nil, fmt.Errorf("Failed to get qingcloud Provider. GetCloudProvider returned %v instead", cloudProvider)
	}

	return qingCloudProvider, nil
}
