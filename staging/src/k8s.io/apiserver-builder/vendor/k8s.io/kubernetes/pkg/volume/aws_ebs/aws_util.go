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

package aws_ebs

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	diskPartitionSuffix = ""
	diskXVDPath         = "/dev/xvd"
	diskXVDPattern      = "/dev/xvd*"
	maxChecks           = 60
	maxRetries          = 10
	checkSleepDuration  = time.Second
	errorSleepDuration  = 5 * time.Second
)

type AWSDiskUtil struct{}

func (util *AWSDiskUtil) DeleteVolume(d *awsElasticBlockStoreDeleter) error {
	cloud, err := getCloudProvider(d.awsElasticBlockStore.plugin.host.GetCloudProvider())
	if err != nil {
		return err
	}

	deleted, err := cloud.DeleteDisk(d.volumeID)
	if err != nil {
		// AWS cloud provider returns volume.deletedVolumeInUseError when
		// necessary, no handling needed here.
		glog.V(2).Infof("Error deleting EBS Disk volume %s: %v", d.volumeID, err)
		return err
	}
	if deleted {
		glog.V(2).Infof("Successfully deleted EBS Disk volume %s", d.volumeID)
	} else {
		glog.V(2).Infof("Successfully deleted EBS Disk volume %s (actually already deleted)", d.volumeID)
	}
	return nil
}

// CreateVolume creates an AWS EBS volume.
// Returns: volumeID, volumeSizeGB, labels, error
func (util *AWSDiskUtil) CreateVolume(c *awsElasticBlockStoreProvisioner) (aws.KubernetesVolumeID, int, map[string]string, error) {
	cloud, err := getCloudProvider(c.awsElasticBlockStore.plugin.host.GetCloudProvider())
	if err != nil {
		return "", 0, nil, err
	}

	// AWS volumes don't have Name field, store the name in Name tag
	var tags map[string]string
	if c.options.CloudTags == nil {
		tags = make(map[string]string)
	} else {
		tags = *c.options.CloudTags
	}
	tags["Name"] = volume.GenerateVolumeName(c.options.ClusterName, c.options.PVName, 255) // AWS tags can have 255 characters

	capacity := c.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	requestBytes := capacity.Value()
	// AWS works with gigabytes, convert to GiB with rounding up
	requestGB := int(volume.RoundUpSize(requestBytes, 1024*1024*1024))
	volumeOptions := &aws.VolumeOptions{
		CapacityGB: requestGB,
		Tags:       tags,
		PVCName:    c.options.PVC.Name,
	}
	// Apply Parameters (case-insensitive). We leave validation of
	// the values to the cloud provider.
	for k, v := range c.options.Parameters {
		switch strings.ToLower(k) {
		case "type":
			volumeOptions.VolumeType = v
		case "zone":
			volumeOptions.AvailabilityZone = v
		case "iopspergb":
			volumeOptions.IOPSPerGB, err = strconv.Atoi(v)
			if err != nil {
				return "", 0, nil, fmt.Errorf("invalid iopsPerGB value %q, must be integer between 1 and 30: %v", v, err)
			}
		case "encrypted":
			volumeOptions.Encrypted, err = strconv.ParseBool(v)
			if err != nil {
				return "", 0, nil, fmt.Errorf("invalid encrypted boolean value %q, must be true or false: %v", v, err)
			}
		case "kmskeyid":
			volumeOptions.KmsKeyId = v
		default:
			return "", 0, nil, fmt.Errorf("invalid option %q for volume plugin %s", k, c.plugin.GetPluginName())
		}
	}

	// TODO: implement PVC.Selector parsing
	if c.options.PVC.Spec.Selector != nil {
		return "", 0, nil, fmt.Errorf("claim.Spec.Selector is not supported for dynamic provisioning on AWS")
	}

	name, err := cloud.CreateDisk(volumeOptions)
	if err != nil {
		glog.V(2).Infof("Error creating EBS Disk volume: %v", err)
		return "", 0, nil, err
	}
	glog.V(2).Infof("Successfully created EBS Disk volume %s", name)

	labels, err := cloud.GetVolumeLabels(name)
	if err != nil {
		// We don't really want to leak the volume here...
		glog.Errorf("error building labels for new EBS volume %q: %v", name, err)
	}

	return name, int(requestGB), labels, nil
}

// Returns the first path that exists, or empty string if none exist.
func verifyDevicePath(devicePaths []string) (string, error) {
	for _, path := range devicePaths {
		if pathExists, err := volumeutil.PathExists(path); err != nil {
			return "", fmt.Errorf("Error checking if path exists: %v", err)
		} else if pathExists {
			return path, nil
		}
	}

	return "", nil
}

// Returns the first path that exists, or empty string if none exist.
func verifyAllPathsRemoved(devicePaths []string) (bool, error) {
	allPathsRemoved := true
	for _, path := range devicePaths {
		if exists, err := volumeutil.PathExists(path); err != nil {
			return false, fmt.Errorf("Error checking if path exists: %v", err)
		} else {
			allPathsRemoved = allPathsRemoved && !exists
		}
	}

	return allPathsRemoved, nil
}

// Returns list of all paths for given EBS mount
// This is more interesting on GCE (where we are able to identify volumes under /dev/disk-by-id)
// Here it is mostly about applying the partition path
func getDiskByIdPaths(partition string, devicePath string) []string {
	devicePaths := []string{}
	if devicePath != "" {
		devicePaths = append(devicePaths, devicePath)
	}

	if partition != "" {
		for i, path := range devicePaths {
			devicePaths[i] = path + diskPartitionSuffix + partition
		}
	}

	return devicePaths
}

// Return cloud provider
func getCloudProvider(cloudProvider cloudprovider.Interface) (*aws.Cloud, error) {
	awsCloudProvider, ok := cloudProvider.(*aws.Cloud)
	if !ok || awsCloudProvider == nil {
		return nil, fmt.Errorf("Failed to get AWS Cloud Provider. GetCloudProvider returned %v instead", cloudProvider)
	}

	return awsCloudProvider, nil
}
