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

package gce_pd

import (
	"fmt"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/cloudprovider"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	diskByIdPath         = "/dev/disk/by-id/"
	diskGooglePrefix     = "google-"
	diskScsiGooglePrefix = "scsi-0Google_PersistentDisk_"
	diskPartitionSuffix  = "-part"
	diskSDPath           = "/dev/sd"
	diskSDPattern        = "/dev/sd*"
	maxChecks            = 60
	maxRetries           = 10
	checkSleepDuration   = time.Second
)

// These variables are modified only in unit tests and should be constant
// otherwise.
var (
	errorSleepDuration time.Duration = 5 * time.Second
)

type GCEDiskUtil struct{}

func (util *GCEDiskUtil) DeleteVolume(d *gcePersistentDiskDeleter) error {
	cloud, err := getCloudProvider(d.gcePersistentDisk.plugin.host.GetCloudProvider())
	if err != nil {
		return err
	}

	if err = cloud.DeleteDisk(d.pdName); err != nil {
		glog.V(2).Infof("Error deleting GCE PD volume %s: %v", d.pdName, err)
		return err
	}
	glog.V(2).Infof("Successfully deleted GCE PD volume %s", d.pdName)
	return nil
}

// CreateVolume creates a GCE PD.
// Returns: volumeID, volumeSizeGB, labels, error
func (gceutil *GCEDiskUtil) CreateVolume(c *gcePersistentDiskProvisioner) (string, int, map[string]string, error) {
	cloud, err := getCloudProvider(c.gcePersistentDisk.plugin.host.GetCloudProvider())
	if err != nil {
		return "", 0, nil, err
	}

	name := volume.GenerateVolumeName(c.options.ClusterName, c.options.PVName, 63) // GCE PD name can have up to 63 characters
	requestBytes := c.options.Capacity.Value()
	// GCE works with gigabytes, convert to GiB with rounding up
	requestGB := volume.RoundUpSize(requestBytes, 1024*1024*1024)

	// Apply Parameters (case-insensitive). We leave validation of
	// the values to the cloud provider.
	diskType := ""
	zone := ""
	for k, v := range c.options.Parameters {
		switch strings.ToLower(k) {
		case "type":
			diskType = v
		case "zone":
			zone = v
		default:
			return "", 0, nil, fmt.Errorf("invalid option %q for volume plugin %s", k, c.plugin.GetPluginName())
		}
	}

	// TODO: implement c.options.ProvisionerSelector parsing
	if c.options.Selector != nil {
		return "", 0, nil, fmt.Errorf("claim.Spec.Selector is not supported for dynamic provisioning on GCE")
	}

	if zone == "" {
		// No zone specified, choose one randomly in the same region as the
		// node is running.
		zones, err := cloud.GetAllZones()
		if err != nil {
			glog.V(2).Infof("error getting zone information from GCE: %v", err)
			return "", 0, nil, err
		}
		zone = volume.ChooseZoneForVolume(zones, c.options.PVCName)
	}

	err = cloud.CreateDisk(name, diskType, zone, int64(requestGB), *c.options.CloudTags)
	if err != nil {
		glog.V(2).Infof("Error creating GCE PD volume: %v", err)
		return "", 0, nil, err
	}
	glog.V(2).Infof("Successfully created GCE PD volume %s", name)

	labels, err := cloud.GetAutoLabelsForPD(name, zone)
	if err != nil {
		// We don't really want to leak the volume here...
		glog.Errorf("error getting labels for volume %q: %v", name, err)
	}

	return name, int(requestGB), labels, nil
}

// Returns the first path that exists, or empty string if none exist.
func verifyDevicePath(devicePaths []string, sdBeforeSet sets.String) (string, error) {
	if err := udevadmChangeToNewDrives(sdBeforeSet); err != nil {
		// udevadm errors should not block disk detachment, log and continue
		glog.Errorf("udevadmChangeToNewDrives failed with: %v", err)
	}

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
		if err := udevadmChangeToDrive(path); err != nil {
			// udevadm errors should not block disk detachment, log and continue
			glog.Errorf("%v", err)
		}
		if exists, err := volumeutil.PathExists(path); err != nil {
			return false, fmt.Errorf("Error checking if path exists: %v", err)
		} else {
			allPathsRemoved = allPathsRemoved && !exists
		}
	}

	return allPathsRemoved, nil
}

// Returns list of all /dev/disk/by-id/* paths for given PD.
func getDiskByIdPaths(pdName string, partition string) []string {
	devicePaths := []string{
		path.Join(diskByIdPath, diskGooglePrefix+pdName),
		path.Join(diskByIdPath, diskScsiGooglePrefix+pdName),
	}

	if partition != "" {
		for i, path := range devicePaths {
			devicePaths[i] = path + diskPartitionSuffix + partition
		}
	}

	return devicePaths
}

// Return cloud provider
func getCloudProvider(cloudProvider cloudprovider.Interface) (*gcecloud.GCECloud, error) {
	var err error
	for numRetries := 0; numRetries < maxRetries; numRetries++ {
		gceCloudProvider, ok := cloudProvider.(*gcecloud.GCECloud)
		if !ok || gceCloudProvider == nil {
			// Retry on error. See issue #11321
			glog.Errorf("Failed to get GCE Cloud Provider. plugin.host.GetCloudProvider returned %v instead", cloudProvider)
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
		return fmt.Errorf("Error filepath.Glob(\"%s\"): %v\r\n", diskSDPattern, err)
	}

	for _, sd := range sdAfter {
		if !sdBeforeSet.Has(sd) {
			return udevadmChangeToDrive(sd)
		}
	}

	return nil
}

// Calls "udevadm trigger --action=change" on the specified drive.
// drivePath must be the the block device path to trigger on, in the format "/dev/sd*", or a symlink to it.
// This is workaround for Issue #7972. Once the underlying issue has been resolved, this may be removed.
func udevadmChangeToDrive(drivePath string) error {
	glog.V(5).Infof("udevadmChangeToDrive: drive=%q", drivePath)

	// Evaluate symlink, if any
	drive, err := filepath.EvalSymlinks(drivePath)
	if err != nil {
		return fmt.Errorf("udevadmChangeToDrive: filepath.EvalSymlinks(%q) failed with %v.", drivePath, err)
	}
	glog.V(5).Infof("udevadmChangeToDrive: symlink path is %q", drive)

	// Check to make sure input is "/dev/sd*"
	if !strings.Contains(drive, diskSDPath) {
		return fmt.Errorf("udevadmChangeToDrive: expected input in the form \"%s\" but drive is %q.", diskSDPattern, drive)
	}

	// Call "udevadm trigger --action=change --property-match=DEVNAME=/dev/sd..."
	_, err = exec.New().Command(
		"udevadm",
		"trigger",
		"--action=change",
		fmt.Sprintf("--property-match=DEVNAME=%s", drive)).CombinedOutput()
	if err != nil {
		return fmt.Errorf("udevadmChangeToDrive: udevadm trigger failed for drive %q with %v.", drive, err)
	}
	return nil
}
