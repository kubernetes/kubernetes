/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"os"
	"path/filepath"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/volume"
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

// Singleton key mutex for keeping attach/detach operations for the same PD atomic
var attachDetachMutex = keymutex.NewKeyMutex()

type AWSDiskUtil struct{}

// Attaches a disk to the current kubelet.
// Mounts the disk to it's global path.
func (diskUtil *AWSDiskUtil) AttachAndMountDisk(b *awsElasticBlockStoreBuilder, globalPDPath string) error {
	glog.V(5).Infof("AttachAndMountDisk(...) called for PD %q. Will block for existing operations, if any. (globalPDPath=%q)\r\n", b.volumeID, globalPDPath)

	// Block execution until any pending detach operations for this PD have completed
	attachDetachMutex.LockKey(b.volumeID)
	defer attachDetachMutex.UnlockKey(b.volumeID)

	glog.V(5).Infof("AttachAndMountDisk(...) called for PD %q. Awake and ready to execute. (globalPDPath=%q)\r\n", b.volumeID, globalPDPath)

	xvdBefore, err := filepath.Glob(diskXVDPattern)
	if err != nil {
		glog.Errorf("Error filepath.Glob(\"%s\"): %v\r\n", diskXVDPattern, err)
	}
	xvdBeforeSet := sets.NewString(xvdBefore...)

	devicePath, err := attachDiskAndVerify(b, xvdBeforeSet)
	if err != nil {
		return err
	}

	// Only mount the PD globally once.
	notMnt, err := b.mounter.IsLikelyNotMountPoint(globalPDPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(globalPDPath, 0750); err != nil {
				return err
			}
			notMnt = true
		} else {
			return err
		}
	}
	options := []string{}
	if b.readOnly {
		options = append(options, "ro")
	}
	if notMnt {
		err = b.diskMounter.FormatAndMount(devicePath, globalPDPath, b.fsType, options)
		if err != nil {
			os.Remove(globalPDPath)
			return err
		}
	}
	return nil
}

// Unmounts the device and detaches the disk from the kubelet's host machine.
func (util *AWSDiskUtil) DetachDisk(c *awsElasticBlockStoreCleaner) error {
	glog.V(5).Infof("DetachDisk(...) for PD %q\r\n", c.volumeID)

	if err := unmountPDAndRemoveGlobalPath(c); err != nil {
		glog.Errorf("Error unmounting PD %q: %v", c.volumeID, err)
	}

	// Detach disk asynchronously so that the kubelet sync loop is not blocked.
	go detachDiskAndVerify(c)
	return nil
}

func (util *AWSDiskUtil) DeleteVolume(d *awsElasticBlockStoreDeleter) error {
	cloud, err := getCloudProvider()
	if err != nil {
		return err
	}

	deleted, err := cloud.DeleteDisk(d.volumeID)
	if err != nil {
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
func (util *AWSDiskUtil) CreateVolume(c *awsElasticBlockStoreProvisioner) (string, int, map[string]string, error) {
	cloud, err := getCloudProvider()
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

	requestBytes := c.options.Capacity.Value()
	// AWS works with gigabytes, convert to GiB with rounding up
	requestGB := int(volume.RoundUpSize(requestBytes, 1024*1024*1024))
	volumeOptions := &aws.VolumeOptions{
		CapacityGB: requestGB,
		Tags:       tags,
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

// Attaches the specified persistent disk device to node, verifies that it is attached, and retries if it fails.
func attachDiskAndVerify(b *awsElasticBlockStoreBuilder, xvdBeforeSet sets.String) (string, error) {
	var awsCloud *aws.AWSCloud
	var attachError error

	for numRetries := 0; numRetries < maxRetries; numRetries++ {
		var err error
		if awsCloud == nil {
			awsCloud, err = getCloudProvider()
			if err != nil || awsCloud == nil {
				// Retry on error. See issue #11321
				glog.Errorf("Error getting AWSCloudProvider while detaching PD %q: %v", b.volumeID, err)
				time.Sleep(errorSleepDuration)
				continue
			}
		}

		if numRetries > 0 {
			glog.Warningf("Retrying attach for EBS Disk %q (retry count=%v).", b.volumeID, numRetries)
		}

		var devicePath string
		devicePath, attachError = awsCloud.AttachDisk(b.volumeID, "", b.readOnly)
		if attachError != nil {
			glog.Errorf("Error attaching PD %q: %v", b.volumeID, attachError)
			time.Sleep(errorSleepDuration)
			continue
		}

		devicePaths := getDiskByIdPaths(b.awsElasticBlockStore, devicePath)

		for numChecks := 0; numChecks < maxChecks; numChecks++ {
			path, err := verifyDevicePath(devicePaths)
			if err != nil {
				// Log error, if any, and continue checking periodically. See issue #11321
				glog.Errorf("Error verifying EBS Disk (%q) is attached: %v", b.volumeID, err)
			} else if path != "" {
				// A device path has successfully been created for the PD
				glog.Infof("Successfully attached EBS Disk %q.", b.volumeID)
				return path, nil
			}

			// Sleep then check again
			glog.V(3).Infof("Waiting for EBS Disk %q to attach.", b.volumeID)
			time.Sleep(checkSleepDuration)
		}
	}

	if attachError != nil {
		return "", fmt.Errorf("Could not attach EBS Disk %q: %v", b.volumeID, attachError)
	}
	return "", fmt.Errorf("Could not attach EBS Disk %q. Timeout waiting for mount paths to be created.", b.volumeID)
}

// Returns the first path that exists, or empty string if none exist.
func verifyDevicePath(devicePaths []string) (string, error) {
	for _, path := range devicePaths {
		if pathExists, err := pathExists(path); err != nil {
			return "", fmt.Errorf("Error checking if path exists: %v", err)
		} else if pathExists {
			return path, nil
		}
	}

	return "", nil
}

// Detaches the specified persistent disk device from node, verifies that it is detached, and retries if it fails.
// This function is intended to be called asynchronously as a go routine.
func detachDiskAndVerify(c *awsElasticBlockStoreCleaner) {
	glog.V(5).Infof("detachDiskAndVerify(...) for pd %q. Will block for pending operations", c.volumeID)
	defer runtime.HandleCrash()

	// Block execution until any pending attach/detach operations for this PD have completed
	attachDetachMutex.LockKey(c.volumeID)
	defer attachDetachMutex.UnlockKey(c.volumeID)

	glog.V(5).Infof("detachDiskAndVerify(...) for pd %q. Awake and ready to execute.", c.volumeID)

	var awsCloud *aws.AWSCloud
	for numRetries := 0; numRetries < maxRetries; numRetries++ {
		var err error
		if awsCloud == nil {
			awsCloud, err = getCloudProvider()
			if err != nil || awsCloud == nil {
				// Retry on error. See issue #11321
				glog.Errorf("Error getting AWSCloudProvider while detaching PD %q: %v", c.volumeID, err)
				time.Sleep(errorSleepDuration)
				continue
			}
		}

		if numRetries > 0 {
			glog.Warningf("Retrying detach for EBS Disk %q (retry count=%v).", c.volumeID, numRetries)
		}

		devicePath, err := awsCloud.DetachDisk(c.volumeID, "")
		if err != nil {
			glog.Errorf("Error detaching PD %q: %v", c.volumeID, err)
			time.Sleep(errorSleepDuration)
			continue
		}

		devicePaths := getDiskByIdPaths(c.awsElasticBlockStore, devicePath)

		for numChecks := 0; numChecks < maxChecks; numChecks++ {
			allPathsRemoved, err := verifyAllPathsRemoved(devicePaths)
			if err != nil {
				// Log error, if any, and continue checking periodically.
				glog.Errorf("Error verifying EBS Disk (%q) is detached: %v", c.volumeID, err)
			} else if allPathsRemoved {
				// All paths to the PD have been successfully removed
				unmountPDAndRemoveGlobalPath(c)
				glog.Infof("Successfully detached EBS Disk %q.", c.volumeID)
				return
			}

			// Sleep then check again
			glog.V(3).Infof("Waiting for EBS Disk %q to detach.", c.volumeID)
			time.Sleep(checkSleepDuration)
		}

	}

	glog.Errorf("Failed to detach EBS Disk %q. One or more mount paths was not removed.", c.volumeID)
}

// Unmount the global PD mount, which should be the only one, and delete it.
func unmountPDAndRemoveGlobalPath(c *awsElasticBlockStoreCleaner) error {
	globalPDPath := makeGlobalPDPath(c.plugin.host, c.volumeID)

	err := c.mounter.Unmount(globalPDPath)
	os.Remove(globalPDPath)
	return err
}

// Returns the first path that exists, or empty string if none exist.
func verifyAllPathsRemoved(devicePaths []string) (bool, error) {
	allPathsRemoved := true
	for _, path := range devicePaths {
		if exists, err := pathExists(path); err != nil {
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
func getDiskByIdPaths(d *awsElasticBlockStore, devicePath string) []string {
	devicePaths := []string{}
	if devicePath != "" {
		devicePaths = append(devicePaths, devicePath)
	}

	if d.partition != "" {
		for i, path := range devicePaths {
			devicePaths[i] = path + diskPartitionSuffix + d.partition
		}
	}

	return devicePaths
}

// Checks if the specified path exists
func pathExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	} else if os.IsNotExist(err) {
		return false, nil
	} else {
		return false, err
	}
}

// Return cloud provider
func getCloudProvider() (*aws.AWSCloud, error) {
	awsCloudProvider, err := cloudprovider.GetCloudProvider("aws", nil)
	if err != nil || awsCloudProvider == nil {
		return nil, err
	}

	// The conversion must be safe otherwise bug in GetCloudProvider()
	return awsCloudProvider.(*aws.AWSCloud), nil
}
