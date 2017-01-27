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

package vsphere_volume

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	maxRetries         = 10
	checkSleepDuration = time.Second
	diskByIDPath       = "/dev/disk/by-id/"
	diskSCSIPrefix     = "wwn-0x"
)

var ErrProbeVolume = errors.New("Error scanning attached volumes")

type VsphereDiskUtil struct{}

func verifyDevicePath(path string) (string, error) {
	if pathExists, err := volumeutil.PathExists(path); err != nil {
		return "", fmt.Errorf("Error checking if path exists: %v", err)
	} else if pathExists {
		return path, nil
	}

	return "", nil
}

// CreateVolume creates a vSphere volume.
func (util *VsphereDiskUtil) CreateVolume(v *vsphereVolumeProvisioner) (vmDiskPath string, volumeSizeKB int, err error) {
	cloud, err := getCloudProvider(v.plugin.host.GetCloudProvider())
	if err != nil {
		return "", 0, err
	}

	capacity := v.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	volSizeBytes := capacity.Value()
	// vSphere works with kilobytes, convert to KiB with rounding up
	volSizeKB := int(volume.RoundUpSize(volSizeBytes, 1024))
	name := volume.GenerateVolumeName(v.options.ClusterName, v.options.PVName, 255)
	volumeOptions := &vsphere.VolumeOptions{
		CapacityKB: volSizeKB,
		Tags:       *v.options.CloudTags,
		Name:       name,
	}

	// Apply Parameters (case-insensitive). We leave validation of
	// the values to the cloud provider.
	for parameter, value := range v.options.Parameters {
		switch strings.ToLower(parameter) {
		case "diskformat":
			volumeOptions.DiskFormat = value
		case "datastore":
			volumeOptions.Datastore = value
		default:
			return "", 0, fmt.Errorf("invalid option %q for volume plugin %s", parameter, v.plugin.GetPluginName())
		}
	}

	// TODO: implement PVC.Selector parsing
	if v.options.PVC.Spec.Selector != nil {
		return "", 0, fmt.Errorf("claim.Spec.Selector is not supported for dynamic provisioning on vSphere")
	}

	vmDiskPath, err = cloud.CreateVolume(volumeOptions)
	if err != nil {
		glog.V(2).Infof("Error creating vsphere volume: %v", err)
		return "", 0, err
	}
	glog.V(2).Infof("Successfully created vsphere volume %s", name)
	return vmDiskPath, volSizeKB, nil
}

// DeleteVolume deletes a vSphere volume.
func (util *VsphereDiskUtil) DeleteVolume(vd *vsphereVolumeDeleter) error {
	cloud, err := getCloudProvider(vd.plugin.host.GetCloudProvider())
	if err != nil {
		return err
	}

	if err = cloud.DeleteVolume(vd.volPath); err != nil {
		glog.V(2).Infof("Error deleting vsphere volume %s: %v", vd.volPath, err)
		return err
	}
	glog.V(2).Infof("Successfully deleted vsphere volume %s", vd.volPath)
	return nil
}

func getVolPathfromDeviceMountPath(deviceMountPath string) string {
	// Assumption: No file or folder is named starting with '[' in datastore
	volPath := deviceMountPath[strings.LastIndex(deviceMountPath, "["):]
	// space between datastore and vmdk name in volumePath is encoded as '\040' when returned by GetMountRefs().
	// volumePath eg: "[local] xxx.vmdk" provided to attach/mount
	// replacing \040 with space to match the actual volumePath
	return strings.Replace(volPath, "\\040", " ", -1)
}

func getCloudProvider(cloud cloudprovider.Interface) (*vsphere.VSphere, error) {
	if cloud == nil {
		glog.Errorf("Cloud provider not initialized properly")
		return nil, errors.New("Cloud provider not initialized properly")
	}

	vs := cloud.(*vsphere.VSphere)
	if vs == nil {
		return nil, errors.New("Invalid cloud provider: expected vSphere")
	}
	return vs, nil
}
