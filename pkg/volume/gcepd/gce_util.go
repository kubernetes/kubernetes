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

package gcepd

import (
	"fmt"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/features"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	utilfile "k8s.io/kubernetes/pkg/util/file"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/utils/exec"
)

const (
	diskByIDPath         = "/dev/disk/by-id/"
	diskGooglePrefix     = "google-"
	diskScsiGooglePrefix = "scsi-0Google_PersistentDisk_"
	diskPartitionSuffix  = "-part"
	diskSDPath           = "/dev/sd"
	diskSDPattern        = "/dev/sd*"
	maxRetries           = 10
	checkSleepDuration   = time.Second
	maxRegionalPDZones   = 2

	// Replication type constants must be lower case.
	replicationTypeNone       = "none"
	replicationTypeRegionalPD = "regional-pd"

	// scsi_id output should be in the form of:
	// 0Google PersistentDisk <disk name>
	scsiPattern = `^0Google\s+PersistentDisk\s+([\S]+)\s*$`
)

var (
	// errorSleepDuration is modified only in unit tests and should be constant
	// otherwise.
	errorSleepDuration = 5 * time.Second

	// regex to parse scsi_id output and extract the serial
	scsiRegex = regexp.MustCompile(scsiPattern)
)

// GCEDiskUtil provides operation for GCE PD
type GCEDiskUtil struct{}

// DeleteVolume deletes a GCE PD
// Returns: error
func (util *GCEDiskUtil) DeleteVolume(d *gcePersistentDiskDeleter) error {
	cloud, err := getCloudProvider(d.gcePersistentDisk.plugin.host.GetCloudProvider())
	if err != nil {
		return err
	}

	if err = cloud.DeleteDisk(d.pdName); err != nil {
		klog.V(2).Infof("Error deleting GCE PD volume %s: %v", d.pdName, err)
		// GCE cloud provider returns volume.deletedVolumeInUseError when
		// necessary, no handling needed here.
		return err
	}
	klog.V(2).Infof("Successfully deleted GCE PD volume %s", d.pdName)
	return nil
}

// CreateVolume creates a GCE PD.
// Returns: gcePDName, volumeSizeGB, labels, fsType, error
func (util *GCEDiskUtil) CreateVolume(c *gcePersistentDiskProvisioner, node *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (string, int, map[string]string, string, error) {
	cloud, err := getCloudProvider(c.gcePersistentDisk.plugin.host.GetCloudProvider())
	if err != nil {
		return "", 0, nil, "", err
	}

	name := volumeutil.GenerateVolumeName(c.options.ClusterName, c.options.PVName, 63) // GCE PD name can have up to 63 characters
	capacity := c.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	// GCE PDs are allocated in chunks of GiBs
	requestGB := volumeutil.RoundUpToGiB(capacity)

	// Apply Parameters.
	// Values for parameter "replication-type" are canonicalized to lower case.
	// Values for other parameters are case-insensitive, and we leave validation of these values
	// to the cloud provider.
	diskType := ""
	configuredZone := ""
	var configuredZones sets.String
	zonePresent := false
	zonesPresent := false
	replicationType := replicationTypeNone
	fstype := ""
	for k, v := range c.options.Parameters {
		switch strings.ToLower(k) {
		case "type":
			diskType = v
		case "zone":
			zonePresent = true
			configuredZone = v
		case "zones":
			zonesPresent = true
			configuredZones, err = volumeutil.ZonesToSet(v)
			if err != nil {
				return "", 0, nil, "", err
			}
		case "replication-type":
			if !utilfeature.DefaultFeatureGate.Enabled(features.GCERegionalPersistentDisk) {
				return "", 0, nil, "",
					fmt.Errorf("the %q option for volume plugin %v is only supported with the %q Kubernetes feature gate enabled",
						k, c.plugin.GetPluginName(), features.GCERegionalPersistentDisk)
			}
			replicationType = strings.ToLower(v)
		case volume.VolumeParameterFSType:
			fstype = v
		default:
			return "", 0, nil, "", fmt.Errorf("invalid option %q for volume plugin %s", k, c.plugin.GetPluginName())
		}
	}

	// TODO: implement PVC.Selector parsing
	if c.options.PVC.Spec.Selector != nil {
		return "", 0, nil, "", fmt.Errorf("claim.Spec.Selector is not supported for dynamic provisioning on GCE")
	}

	var activezones sets.String
	activezones, err = cloud.GetAllCurrentZones()
	if err != nil {
		return "", 0, nil, "", err
	}

	switch replicationType {
	case replicationTypeRegionalPD:
		selectedZones, err := volumeutil.SelectZonesForVolume(zonePresent, zonesPresent, configuredZone, configuredZones, activezones, node, allowedTopologies, c.options.PVC.Name, maxRegionalPDZones)
		if err != nil {
			klog.V(2).Infof("Error selecting zones for regional GCE PD volume: %v", err)
			return "", 0, nil, "", err
		}
		if err = cloud.CreateRegionalDisk(
			name,
			diskType,
			selectedZones,
			int64(requestGB),
			*c.options.CloudTags); err != nil {
			klog.V(2).Infof("Error creating regional GCE PD volume: %v", err)
			return "", 0, nil, "", err
		}
		klog.V(2).Infof("Successfully created Regional GCE PD volume %s", name)

	case replicationTypeNone:
		selectedZone, err := volumeutil.SelectZoneForVolume(zonePresent, zonesPresent, configuredZone, configuredZones, activezones, node, allowedTopologies, c.options.PVC.Name)
		if err != nil {
			return "", 0, nil, "", err
		}
		if err := cloud.CreateDisk(
			name,
			diskType,
			selectedZone,
			int64(requestGB),
			*c.options.CloudTags); err != nil {
			klog.V(2).Infof("Error creating single-zone GCE PD volume: %v", err)
			return "", 0, nil, "", err
		}
		klog.V(2).Infof("Successfully created single-zone GCE PD volume %s", name)

	default:
		return "", 0, nil, "", fmt.Errorf("replication-type of '%s' is not supported", replicationType)
	}

	labels, err := cloud.GetAutoLabelsForPD(name, "" /* zone */)
	if err != nil {
		// We don't really want to leak the volume here...
		klog.Errorf("error getting labels for volume %q: %v", name, err)
	}

	return name, int(requestGB), labels, fstype, nil
}

// Returns the first path that exists, or empty string if none exist.
func verifyDevicePath(devicePaths []string, sdBeforeSet sets.String, diskName string) (string, error) {
	if err := udevadmChangeToNewDrives(sdBeforeSet); err != nil {
		// It's possible udevadm was called on other disks so it should not block this
		// call. If it did fail on this disk, then the devicePath will either
		// not exist or be wrong. If it's wrong, then the scsi_id check below will fail.
		klog.Errorf("udevadmChangeToNewDrives failed with: %v", err)
	}

	for _, path := range devicePaths {
		if pathExists, err := volumeutil.PathExists(path); err != nil {
			return "", fmt.Errorf("Error checking if path exists: %v", err)
		} else if pathExists {
			// validate that the path actually resolves to the correct disk
			serial, err := getScsiSerial(path, diskName)
			if err != nil {
				return "", fmt.Errorf("failed to get scsi serial %v", err)
			}
			if serial != diskName {
				// The device link is not pointing to the correct device
				// Trigger udev on this device to try to fix the link
				if udevErr := udevadmChangeToDrive(path); udevErr != nil {
					klog.Errorf("udevadmChangeToDrive %q failed with: %v", path, err)
				}

				// Return error to retry WaitForAttach and verifyDevicePath
				return "", fmt.Errorf("scsi_id serial %q for device %q doesn't match disk %q", serial, path, diskName)
			}
			// The device link is correct
			return path, nil
		}
	}

	return "", nil
}

// Calls scsi_id on the given devicePath to get the serial number reported by that device.
func getScsiSerial(devicePath, diskName string) (string, error) {
	exists, err := utilfile.FileExists("/lib/udev/scsi_id")
	if err != nil {
		return "", fmt.Errorf("failed to check scsi_id existence: %v", err)
	}

	if !exists {
		klog.V(6).Infof("scsi_id doesn't exist; skipping check for %v", devicePath)
		return diskName, nil
	}

	out, err := exec.New().Command(
		"/lib/udev/scsi_id",
		"--page=0x83",
		"--whitelisted",
		fmt.Sprintf("--device=%v", devicePath)).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("scsi_id failed for device %q with %v", devicePath, err)
	}

	return parseScsiSerial(string(out))
}

// Parse the output returned by scsi_id and extract the serial number
func parseScsiSerial(output string) (string, error) {
	substrings := scsiRegex.FindStringSubmatch(output)
	if substrings == nil {
		return "", fmt.Errorf("scsi_id output cannot be parsed: %q", output)
	}

	return substrings[1], nil
}

// Returns list of all /dev/disk/by-id/* paths for given PD.
func getDiskByIDPaths(pdName string, partition string) []string {
	devicePaths := []string{
		path.Join(diskByIDPath, diskGooglePrefix+pdName),
		path.Join(diskByIDPath, diskScsiGooglePrefix+pdName),
	}

	if partition != "" {
		for i, path := range devicePaths {
			devicePaths[i] = path + diskPartitionSuffix + partition
		}
	}

	return devicePaths
}

// Return cloud provider
func getCloudProvider(cloudProvider cloudprovider.Interface) (*gcecloud.Cloud, error) {
	var err error
	for numRetries := 0; numRetries < maxRetries; numRetries++ {
		gceCloudProvider, ok := cloudProvider.(*gcecloud.Cloud)
		if !ok || gceCloudProvider == nil {
			// Retry on error. See issue #11321
			klog.Errorf("Failed to get GCE Cloud Provider. plugin.host.GetCloudProvider returned %v instead", cloudProvider)
			time.Sleep(errorSleepDuration)
			continue
		}

		return gceCloudProvider, nil
	}

	return nil, fmt.Errorf("Failed to get GCE GCECloudProvider with error %v", err)
}

// Triggers the application of udev rules by calling "udevadm trigger
// --action=change" for newly created "/dev/sd*" drives (exist only in
// after set). This is workaround for Issue #7972. Once the underlying
// issue has been resolved, this may be removed.
func udevadmChangeToNewDrives(sdBeforeSet sets.String) error {
	sdAfter, err := filepath.Glob(diskSDPattern)
	if err != nil {
		return fmt.Errorf("error filepath.Glob(\"%s\"): %v\r", diskSDPattern, err)
	}

	for _, sd := range sdAfter {
		if !sdBeforeSet.Has(sd) {
			return udevadmChangeToDrive(sd)
		}
	}

	return nil
}

// Calls "udevadm trigger --action=change" on the specified drive.
// drivePath must be the block device path to trigger on, in the format "/dev/sd*", or a symlink to it.
// This is workaround for Issue #7972. Once the underlying issue has been resolved, this may be removed.
func udevadmChangeToDrive(drivePath string) error {
	klog.V(5).Infof("udevadmChangeToDrive: drive=%q", drivePath)

	// Evaluate symlink, if any
	drive, err := filepath.EvalSymlinks(drivePath)
	if err != nil {
		return fmt.Errorf("udevadmChangeToDrive: filepath.EvalSymlinks(%q) failed with %v", drivePath, err)
	}
	klog.V(5).Infof("udevadmChangeToDrive: symlink path is %q", drive)

	// Check to make sure input is "/dev/sd*"
	if !strings.Contains(drive, diskSDPath) {
		return fmt.Errorf("udevadmChangeToDrive: expected input in the form \"%s\" but drive is %q", diskSDPattern, drive)
	}

	// Call "udevadm trigger --action=change --property-match=DEVNAME=/dev/sd..."
	_, err = exec.New().Command(
		"udevadm",
		"trigger",
		"--action=change",
		fmt.Sprintf("--property-match=DEVNAME=%s", drive)).CombinedOutput()
	if err != nil {
		return fmt.Errorf("udevadmChangeToDrive: udevadm trigger failed for drive %q with %v", drive, err)
	}
	return nil
}

// Checks whether the given GCE PD volume spec is associated with a regional PD.
func isRegionalPD(spec *volume.Spec) bool {
	if spec.PersistentVolume != nil {
		zonesLabel := spec.PersistentVolume.Labels[kubeletapis.LabelZoneFailureDomain]
		zones := strings.Split(zonesLabel, kubeletapis.LabelMultiZoneDelimiter)
		return len(zones) > 1
	}
	return false
}
