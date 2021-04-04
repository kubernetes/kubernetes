// +build !providerless
// +build linux

/*
Copyright 2019 The Kubernetes Authors.

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

package awsebs

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/klog/v2"
	"k8s.io/legacy-cloud-providers/aws"
)

const (
	diskPartitionSuffix     = ""
	nvmeDiskPartitionSuffix = "p"
)

func (attacher *awsElasticBlockStoreAttacher) getDevicePath(volumeID, partition, devicePath string) (string, error) {
	devicePaths := getDiskByIDPaths(aws.KubernetesVolumeID(volumeID), partition, devicePath)
	return verifyDevicePath(devicePaths)
}

// findNvmeVolume looks for the nvme volume with the specified name
// It follows the symlink (if it exists) and returns the absolute path to the device
func findNvmeVolume(findName string) (device string, err error) {
	p := filepath.Join("/dev/disk/by-id/", findName)
	stat, err := os.Lstat(p)
	if err != nil {
		if os.IsNotExist(err) {
			klog.V(6).Infof("nvme path not found %q", p)
			return "", nil
		}
		return "", fmt.Errorf("error getting stat of %q: %v", p, err)
	}

	if stat.Mode()&os.ModeSymlink != os.ModeSymlink {
		klog.Warningf("nvme file %q found, but was not a symlink", p)
		return "", nil
	}

	// Find the target, resolving to an absolute path
	// For example, /dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_vol0fab1d5e3f72a5e23 -> ../../nvme2n1
	resolved, err := filepath.EvalSymlinks(p)
	if err != nil {
		return "", fmt.Errorf("error reading target of symlink %q: %v", p, err)
	}

	if !strings.HasPrefix(resolved, "/dev") {
		return "", fmt.Errorf("resolved symlink for %q was unexpected: %q", p, resolved)
	}

	return resolved, nil
}

// Returns list of all paths for given EBS mount
// This is more interesting on GCE (where we are able to identify volumes under /dev/disk-by-id)
// Here it is mostly about applying the partition path
func getDiskByIDPaths(volumeID aws.KubernetesVolumeID, partition string, devicePath string) []string {
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
		klog.Warningf("error mapping volume %q to AWS volume: %v", volumeID, err)
	} else {
		// This is the magic name on which AWS presents NVME devices under /dev/disk/by-id/
		// For example, vol-0fab1d5e3f72a5e23 creates a symlink at /dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_vol0fab1d5e3f72a5e23
		nvmeName := "nvme-Amazon_Elastic_Block_Store_" + strings.Replace(string(awsVolumeID), "-", "", -1)
		nvmePath, err := findNvmeVolume(nvmeName)
		if err != nil {
			klog.Warningf("error looking for nvme volume %q: %v", volumeID, err)
		} else if nvmePath != "" {
			if partition != "" {
				nvmePath = nvmePath + nvmeDiskPartitionSuffix + partition
			}
			devicePaths = append(devicePaths, nvmePath)
		}
	}

	return devicePaths
}
