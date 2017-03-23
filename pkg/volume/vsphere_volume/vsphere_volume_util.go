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
	"strconv"
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
	diskformat         = "diskformat"

	Policy_HostFailuresToTolerate = "hostfailurestotolerate"
	Policy_ForceProvisioning      = "forceprovisioning"
	Policy_CacheReservation       = "cachereservation"
	Policy_DiskStripes            = "diskstripes"
	Policy_ObjectSpaceReservation = "objectspacereservation"
	Policy_IopsLimit              = "iopslimit"
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
		case diskformat:
			volumeOptions.DiskFormat = value
		case "datastore":
			volumeOptions.Datastore = value
		case Policy_HostFailuresToTolerate:
			if !validateVSANCapability(Policy_HostFailuresToTolerate, value) {
				return "", 0, fmt.Errorf(`Invalid value for hostFailuresToTolerate in volume plugin %s. 
				The default value is 1, minimum value is 0 and maximum value is 3.`, v.plugin.GetPluginName())
			}
			volumeOptions.StorageProfileData += " (\"hostFailuresToTolerate\" i" + value + ")"
		case Policy_ForceProvisioning:
			if !validateVSANCapability(Policy_ForceProvisioning, value) {
				return "", 0, fmt.Errorf(`Invalid value for forceProvisioning in volume plugin %s. 
				The value can be either 0 or 1.`, v.plugin.GetPluginName())
			}
			volumeOptions.StorageProfileData += " (\"forceProvisioning\" i" + value + ")"
		case Policy_CacheReservation:
			if !validateVSANCapability(Policy_CacheReservation, value) {
				return "", 0, fmt.Errorf(`Invalid value for cacheReservation in volume plugin %s.
				The minimum percentage is 0 and maximum percentage is 100.`, v.plugin.GetPluginName())
			}
			intVal, _ := strconv.Atoi(value)
			volumeOptions.StorageProfileData += " (\"cacheReservation\" i" + strconv.Itoa(intVal*10000) + ")"
		case Policy_DiskStripes:
			if !validateVSANCapability(Policy_DiskStripes, value) {
				return "", 0, fmt.Errorf(`Invalid value for diskStripes in volume plugin %s. 
				The minimum value is 1 and maximum value is 12.`, v.plugin.GetPluginName())
			}
			volumeOptions.StorageProfileData += " (\"stripeWidth\" i" + value + ")"
		case Policy_ObjectSpaceReservation:
			if !validateVSANCapability(Policy_ObjectSpaceReservation, value) {
				return "", 0, fmt.Errorf(`Invalid value for ObjectSpaceReservation in volume plugin %s. 
				The minimum percentage is 0 and maximum percentage is 100.`, v.plugin.GetPluginName())
			}
			volumeOptions.StorageProfileData += " (\"proportionalCapacity\" i" + value + ")"
		case Policy_IopsLimit:
			if !validateVSANCapability(Policy_IopsLimit, value) {
				return "", 0, fmt.Errorf(`Invalid value for iopsLimit in volume plugin %s. 
				The value should be greater than 0.`, v.plugin.GetPluginName())
			}
			volumeOptions.StorageProfileData += " (\"iopsLimit\" i" + value + ")"
		default:
			return "", 0, fmt.Errorf("invalid option %q for volume plugin %s", parameter, v.plugin.GetPluginName())
		}
	}

	if volumeOptions.StorageProfileData != "" {
		volumeOptions.StorageProfileData = "(" + volumeOptions.StorageProfileData + ")"
	}
	glog.V(1).Infof("StorageProfileData in vsphere volume %q", volumeOptions.StorageProfileData)
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

// Validate the capability requirement for the user specified policy attributes.
func validateVSANCapability(capabilityName string, capabilityValue string) bool {
	switch strings.ToLower(capabilityName) {
	case Policy_HostFailuresToTolerate:
		capabilityIntVal, ok := verifyCapabilityValueIsInteger(capabilityValue)
		if ok && (capabilityIntVal >= 0 && capabilityIntVal <= 3) {
			return true
		}
	case Policy_ForceProvisioning:
		capabilityIntVal, ok := verifyCapabilityValueIsInteger(capabilityValue)
		if ok && (capabilityIntVal == 0 || capabilityIntVal == 1) {
			return true
		}
	case Policy_CacheReservation:
		capabilityIntVal, ok := verifyCapabilityValueIsInteger(capabilityValue)
		if ok && (capabilityIntVal >= 0 && capabilityIntVal <= 100) {
			return true
		}
	case Policy_DiskStripes:
		capabilityIntVal, ok := verifyCapabilityValueIsInteger(capabilityValue)
		if ok && (capabilityIntVal >= 1 && capabilityIntVal <= 12) {
			return true
		}
	// Need to check
	case Policy_ObjectSpaceReservation:
		capabilityIntVal, ok := verifyCapabilityValueIsInteger(capabilityValue)
		if ok && (capabilityIntVal >= 0 || capabilityIntVal <= 100) {
			return true
		}
	case Policy_IopsLimit:
		capabilityIntVal, ok := verifyCapabilityValueIsInteger(capabilityValue)
		if ok && (capabilityIntVal >= 0) {
			return true
		}
	}
	return false
}

// Verify if the capability value is of type integer.
func verifyCapabilityValueIsInteger(capabilityValue string) (int, bool) {
	i, err := strconv.Atoi(capabilityValue)
	if err != nil {
		return -1, false
	}
	return i, true
}
