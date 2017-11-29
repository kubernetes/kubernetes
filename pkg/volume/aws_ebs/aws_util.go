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
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/aws_ebs/nvme"
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
func (util *AWSDiskUtil) CreateVolume(c *awsElasticBlockStoreProvisioner) (aws.KubernetesVolumeID, int, map[string]string, string, error) {
	cloud, err := getCloudProvider(c.awsElasticBlockStore.plugin.host.GetCloudProvider())
	if err != nil {
		return "", 0, nil, "", err
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
	fstype := ""
	// Apply Parameters (case-insensitive). We leave validation of
	// the values to the cloud provider.
	volumeOptions.ZonePresent = false
	volumeOptions.ZonesPresent = false
	for k, v := range c.options.Parameters {
		switch strings.ToLower(k) {
		case "type":
			volumeOptions.VolumeType = v
		case "zone":
			volumeOptions.ZonePresent = true
			volumeOptions.AvailabilityZone = v
		case "zones":
			volumeOptions.ZonesPresent = true
			volumeOptions.AvailabilityZones = v
		case "iopspergb":
			volumeOptions.IOPSPerGB, err = strconv.Atoi(v)
			if err != nil {
				return "", 0, nil, "", fmt.Errorf("invalid iopsPerGB value %q, must be integer between 1 and 30: %v", v, err)
			}
		case "encrypted":
			volumeOptions.Encrypted, err = strconv.ParseBool(v)
			if err != nil {
				return "", 0, nil, "", fmt.Errorf("invalid encrypted boolean value %q, must be true or false: %v", v, err)
			}
		case "kmskeyid":
			volumeOptions.KmsKeyId = v
		case volume.VolumeParameterFSType:
			fstype = v
		default:
			return "", 0, nil, "", fmt.Errorf("invalid option %q for volume plugin %s", k, c.plugin.GetPluginName())
		}
	}

	if volumeOptions.ZonePresent && volumeOptions.ZonesPresent {
		return "", 0, nil, "", fmt.Errorf("both zone and zones StorageClass parameters must not be used at the same time")
	}

	// TODO: implement PVC.Selector parsing
	if c.options.PVC.Spec.Selector != nil {
		return "", 0, nil, "", fmt.Errorf("claim.Spec.Selector is not supported for dynamic provisioning on AWS")
	}

	name, err := cloud.CreateDisk(volumeOptions)
	if err != nil {
		glog.V(2).Infof("Error creating EBS Disk volume: %v", err)
		return "", 0, nil, "", err
	}
	glog.V(2).Infof("Successfully created EBS Disk volume %s", name)

	labels, err := cloud.GetVolumeLabels(name)
	if err != nil {
		// We don't really want to leak the volume here...
		glog.Errorf("error building labels for new EBS volume %q: %v", name, err)
	}

	return name, int(requestGB), labels, fstype, nil
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
func getDiskByIdPaths(volumeID aws.KubernetesVolumeID, partition string, devicePath string) []string {
	devicePaths := []string{}
	if devicePath != "" {
		devicePaths = append(devicePaths, devicePath)
	}

	if partition != "" {
		for i, path := range devicePaths {
			devicePaths[i] = path + diskPartitionSuffix + partition
		}
	}

	// We need to find NVME volumes, which are mounted on a "random" nvme path ("/dev/nvme0n1"),
	// and we have to get the volume id from the nvme interface
	awsVolumeID, err := volumeID.MapToAWSVolumeID()
	if err != nil {
		glog.Warningf("error mapping volume %q to AWS volume: %v", volumeID, err)
	} else {
		nvmeName := strings.Replace(string(awsVolumeID), "-", "", -1)
		nvmePath, err := findNvmeVolume(nvmeName)
		if err != nil {
			glog.Warningf("error looking for nvme volume %q: %v", volumeID, err)
		} else if nvmePath != "" {
			devicePaths = append(devicePaths, nvmePath)
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

func findNvmeVolume(findName string) (device string, err error) {
	devices, err := ioutil.ReadDir("/dev")
	if err != nil {
		return "", fmt.Errorf("error listing /dev directory: %v", err)
	}
	for _, device := range devices {
		name := device.Name()
		if !strings.HasPrefix(name, "nvme") {
			continue
		}
		glog.V(6).Infof("found nvme %q", name)

		// Skip partition devices
		{
			num := 0
			ns := 0
			partition := 0
			tokens, err := fmt.Sscanf(name, "nvme%dn%dp%d", &num, &ns, &partition)
			if err == nil && tokens == 3 {
				glog.V(4).Infof("skipping nvme partition device %q", name)
				continue
			}
		}

		p := filepath.Join("/dev", name)

		// Skip non-block devices
		{
			s, err := os.Stat(p)
			if err != nil {
				glog.Warningf("ignoring error doing stat on %q: %v", p, err)
				continue
			}
			mode := s.Mode()
			if (mode&os.ModeDevice == 0) || (mode&os.ModeCharDevice != 0) {
				glog.V(2).Infof("ignoring nvme device %q because of mode %v", p, s.Mode())
				continue
			}
			glog.V(6).Infof("%q had mode %v", p, mode)
		}

		glog.V(4).Infof("found nvme candidate %q", p)

		info, found, err := nvme.Identify(p)
		if err != nil {
			glog.Warningf("ignoring error identifying nvme device %q: %v", p, err)
			continue
		}
		if !found {
			glog.Warningf("identification not supported for nvme device %q", p)
			continue
		}

		glog.V(4).Infof("got info for %q: %q", p, info)
		if info != "" && findName == info {
			glog.Infof("found matching nvme volume %q at %q", info, p)
			return filepath.Join("/dev", name), nil
		}
	}

	return "", nil
}
