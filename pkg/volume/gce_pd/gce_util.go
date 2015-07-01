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

package gce_pd

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/gce"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/operationmanager"
	"github.com/golang/glog"
)

const (
	diskByIdPath         = "/dev/disk/by-id/"
	diskGooglePrefix     = "google-"
	diskScsiGooglePrefix = "scsi-0Google_PersistentDisk_"
	diskPartitionSuffix  = "-part"
	diskSDPath           = "/dev/sd"
	diskSDPattern        = "/dev/sd*"
	maxChecks            = 10
	maxRetries           = 10
	checkSleepDuration   = time.Second
)

// Singleton operation manager for managing detach clean up go routines
var detachCleanupManager = operationmanager.NewOperationManager()

type GCEDiskUtil struct{}

// Attaches a disk specified by a volume.GCEPersistentDisk to the current kubelet.
// Mounts the disk to it's global path.
func (diskUtil *GCEDiskUtil) AttachAndMountDisk(pd *gcePersistentDisk, globalPDPath string) error {
	glog.V(5).Infof("AttachAndMountDisk(pd, %q) where pd is %#v\r\n", globalPDPath, pd)
	// Terminate any in progress verify detach go routines, this will block until the goroutine is ready to exit because the channel is unbuffered
	detachCleanupManager.Send(pd.pdName, true)
	sdBefore, err := filepath.Glob(diskSDPattern)
	if err != nil {
		glog.Errorf("Error filepath.Glob(\"%s\"): %v\r\n", diskSDPattern, err)
	}
	sdBeforeSet := util.NewStringSet(sdBefore...)

	gce, err := cloudprovider.GetCloudProvider("gce", nil)
	if err != nil {
		return err
	}

	if err := gce.(*gce_cloud.GCECloud).AttachDisk(pd.pdName, pd.readOnly); err != nil {
		return err
	}

	devicePath, err := verifyAttached(pd, sdBeforeSet, gce)
	if err != nil {
		return err
	}

	// Only mount the PD globally once.
	mountpoint, err := pd.mounter.IsMountPoint(globalPDPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(globalPDPath, 0750); err != nil {
				return err
			}
			mountpoint = false
		} else {
			return err
		}
	}
	options := []string{}
	if pd.readOnly {
		options = append(options, "ro")
	}
	if !mountpoint {
		err = pd.diskMounter.Mount(devicePath, globalPDPath, pd.fsType, options)
		if err != nil {
			os.Remove(globalPDPath)
			return err
		}
	}
	return nil
}

// Unmounts the device and detaches the disk from the kubelet's host machine.
func (util *GCEDiskUtil) DetachDisk(pd *gcePersistentDisk) error {
	// Unmount the global PD mount, which should be the only one.
	globalPDPath := makeGlobalPDName(pd.plugin.host, pd.pdName)
	glog.V(5).Infof("DetachDisk(pd) where pd is %#v and the globalPDPath is %q\r\n", pd, globalPDPath)

	// Terminate any in progress verify detach go routines, this will block until the goroutine is ready to exit because the channel is unbuffered
	detachCleanupManager.Send(pd.pdName, true)

	if err := pd.mounter.Unmount(globalPDPath); err != nil {
		return err
	}
	if err := os.Remove(globalPDPath); err != nil {
		return err
	}
	// Detach the disk
	gce, err := cloudprovider.GetCloudProvider("gce", nil)
	if err != nil {
		return err
	}
	if err := gce.(*gce_cloud.GCECloud).DetachDisk(pd.pdName); err != nil {
		return err
	}

	// Verify disk detached, retry if needed.
	go verifyDetached(pd, gce)
	return nil
}

// Verifys the disk device to be created has been succesffully attached, and retries if it fails.
func verifyAttached(pd *gcePersistentDisk, sdBeforeSet util.StringSet, gce cloudprovider.Interface) (string, error) {
	devicePaths := getDiskByIdPaths(pd)
	for numRetries := 0; numRetries < maxRetries; numRetries++ {
		for numChecks := 0; numChecks < maxChecks; numChecks++ {
			if err := udevadmChangeToNewDrives(sdBeforeSet); err != nil {
				// udevadm errors should not block disk attachment, log and continue
				glog.Errorf("%v", err)
			}

			for _, path := range devicePaths {
				if pathExists, err := pathExists(path); err != nil {
					return "", err
				} else if pathExists {
					// A device path has succesfully been created for the PD
					glog.V(5).Infof("Succesfully attached GCE PD %q.", pd.pdName)
					return path, nil
				}
			}

			// Sleep then check again
			glog.V(5).Infof("Waiting for GCE PD %q to attach.", pd.pdName)
			time.Sleep(checkSleepDuration)
		}

		// Try attaching the disk again
		glog.Warningf("Timed out waiting for GCE PD %q to attach. Retrying attach.", pd.pdName)
		if err := gce.(*gce_cloud.GCECloud).AttachDisk(pd.pdName, pd.readOnly); err != nil {
			return "", err
		}
	}

	return "", fmt.Errorf("Could not attach GCE PD %q. Timeout waiting for mount paths to be created.", pd.pdName)
}

// Veify the specified persistent disk device has been succesfully detached, and retries if it fails.
// This function is intended to be called asynchronously as a go routine.
func verifyDetached(pd *gcePersistentDisk, gce cloudprovider.Interface) {
	defer util.HandleCrash()

	// Setting bufferSize to 0 so that when senders send, they are blocked until we recieve. This avoids the need to have a separate exit check.
	ch, err := detachCleanupManager.Start(pd.pdName, 0 /* bufferSize */)
	if err != nil {
		glog.Errorf("Error adding %q to detachCleanupManager: %v", pd.pdName, err)
		return
	}
	defer detachCleanupManager.Close(pd.pdName)

	devicePaths := getDiskByIdPaths(pd)
	for numRetries := 0; numRetries < maxRetries; numRetries++ {
		for numChecks := 0; numChecks < maxChecks; numChecks++ {
			select {
			case <-ch:
				glog.Warningf("Terminating GCE PD %q detach verification. Another attach/detach call was made for this PD.", pd.pdName)
				return
			default:
				allPathsRemoved := true
				for _, path := range devicePaths {
					if err := udevadmChangeToDrive(path); err != nil {
						// udevadm errors should not block disk detachment, log and continue
						glog.Errorf("%v", err)
					}
					if exists, err := pathExists(path); err != nil {
						glog.Errorf("Error check path: %v", err)
						return
					} else {
						allPathsRemoved = allPathsRemoved && !exists
					}
				}
				if allPathsRemoved {
					// All paths to the PD have been succefully removed
					glog.V(5).Infof("Succesfully detached GCE PD %q.", pd.pdName)
					return
				}

				// Sleep then check again
				glog.V(5).Infof("Waiting for GCE PD %q to detach.", pd.pdName)
				time.Sleep(checkSleepDuration)
			}
		}

		// Try detaching disk again
		glog.Warningf("Timed out waiting for GCE PD %q to detach. Retrying detach.", pd.pdName)
		if err := gce.(*gce_cloud.GCECloud).DetachDisk(pd.pdName); err != nil {
			glog.Errorf("Error on retry detach PD %q: %v", pd.pdName, err)
			return
		}
	}

	glog.Errorf("Could not detach GCE PD %q. One or more mount paths was not removed.", pd.pdName)
}

// Returns list of all /dev/disk/by-id/* paths for given PD.
func getDiskByIdPaths(pd *gcePersistentDisk) []string {
	devicePaths := []string{
		path.Join(diskByIdPath, diskGooglePrefix+pd.pdName),
		path.Join(diskByIdPath, diskScsiGooglePrefix+pd.pdName),
	}

	if pd.partition != "" {
		for i, path := range devicePaths {
			devicePaths[i] = path + diskPartitionSuffix + pd.partition
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

// Calls "udevadm trigger --action=change" for newly created "/dev/sd*" drives (exist only in after set).
// This is workaround for Issue #7972. Once the underlying issue has been resolved, this may be removed.
func udevadmChangeToNewDrives(sdBeforeSet util.StringSet) error {
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

// safe_format_and_mount is a utility script on GCE VMs that probes a persistent disk, and if
// necessary formats it before mounting it.
// This eliminates the necesisty to format a PD before it is used with a Pod on GCE.
// TODO: port this script into Go and use it for all Linux platforms
type gceSafeFormatAndMount struct {
	mount.Interface
	runner exec.Interface
}

// uses /usr/share/google/safe_format_and_mount to optionally mount, and format a disk
func (mounter *gceSafeFormatAndMount) Mount(source string, target string, fstype string, options []string) error {
	// Don't attempt to format if mounting as readonly. Go straight to mounting.
	for _, option := range options {
		if option == "ro" {
			return mounter.Interface.Mount(source, target, fstype, options)
		}
	}
	args := []string{}
	// ext4 is the default for safe_format_and_mount
	if len(fstype) > 0 && fstype != "ext4" {
		args = append(args, "-m", fmt.Sprintf("mkfs.%s", fstype))
	}
	args = append(args, options...)
	args = append(args, source, target)
	glog.V(5).Infof("exec-ing: /usr/share/google/safe_format_and_mount %v", args)
	cmd := mounter.runner.Command("/usr/share/google/safe_format_and_mount", args...)
	dataOut, err := cmd.CombinedOutput()
	if err != nil {
		glog.V(5).Infof("error running /usr/share/google/safe_format_and_mount\n%s", string(dataOut))
	}
	return err
}
