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
	"io/ioutil"
	"os"
	"path"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	maxRetries = 10
)

var ErrProbeVolume = errors.New("Error scanning attached volumes")

// Singleton key mutex for keeping attach/detach operations for the same PD atomic
var attachDetachMutex = keymutex.NewKeyMutex()

type VsphereDiskUtil struct{}

// Attaches a disk to the current kubelet.
// Mounts the disk to it's global path.
func (util *VsphereDiskUtil) AttachDisk(vm *vsphereVolumeMounter, globalPDPath string) error {
	options := []string{}

	// Block execution until any pending attach/detach operations for this PD have completed
	attachDetachMutex.LockKey(vm.volPath)
	defer attachDetachMutex.UnlockKey(vm.volPath)

	cloud, err := vm.plugin.getCloudProvider()
	if err != nil {
		return err
	}

	diskID, diskUUID, attachError := cloud.AttachDisk(vm.volPath, "")
	if attachError != nil {
		return attachError
	} else if diskUUID == "" {
		return errors.New("Disk UUID has no value")
	}

	// diskID for detach Disk
	vm.diskID = diskID

	var devicePath string
	numTries := 0
	for {
		devicePath = verifyDevicePath(diskUUID)

		_, err := os.Stat(devicePath)
		if err == nil {
			break
		}
		if err != nil && !os.IsNotExist(err) {
			return err
		}
		numTries++
		if numTries == maxRetries {
			return errors.New("Could not attach disk: Timeout after 60s")
		}
		time.Sleep(time.Second * 60)
	}

	notMnt, err := vm.mounter.IsLikelyNotMountPoint(globalPDPath)
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
	if notMnt {
		err = vm.diskMounter.FormatAndMount(devicePath, globalPDPath, vm.fsType, options)
		if err != nil {
			os.Remove(globalPDPath)
			return err
		}
		glog.V(2).Infof("Safe mount successful: %q\n", devicePath)
	}
	return nil
}

func verifyDevicePath(diskUUID string) string {
	files, _ := ioutil.ReadDir("/dev/disk/by-id/")
	for _, f := range files {
		// TODO: should support other controllers
		if strings.Contains(f.Name(), "scsi-") {
			devID := f.Name()[len("scsi-"):len(f.Name())]
			if strings.Contains(devID, diskUUID) {
				glog.V(4).Infof("Found disk attached as %q; full devicepath: %s\n", f.Name(), path.Join("/dev/disk/by-id/", f.Name()))
				return path.Join("/dev/disk/by-id/", f.Name())
			}
		}
	}
	glog.Warningf("Failed to find device for the diskid: %q\n", diskUUID)
	return ""
}

// Unmounts the device and detaches the disk from the kubelet's host machine.
func (util *VsphereDiskUtil) DetachDisk(vu *vsphereVolumeUnmounter) error {

	// Block execution until any pending attach/detach operations for this PD have completed
	attachDetachMutex.LockKey(vu.volPath)
	defer attachDetachMutex.UnlockKey(vu.volPath)

	globalPDPath := makeGlobalPDPath(vu.plugin.host, vu.volPath)
	if err := vu.mounter.Unmount(globalPDPath); err != nil {
		return err
	}
	if err := os.Remove(globalPDPath); err != nil {
		return err
	}
	glog.V(2).Infof("Successfully unmounted main device: %s\n", globalPDPath)

	cloud, err := vu.plugin.getCloudProvider()
	if err != nil {
		return err
	}

	if err = cloud.DetachDisk(vu.volPath, ""); err != nil {
		return err
	}
	glog.V(2).Infof("Successfully detached vSphere volume %s", vu.volPath)
	return nil
}

// CreateVolume creates a vSphere volume.
func (util *VsphereDiskUtil) CreateVolume(v *vsphereVolumeProvisioner) (vmDiskPath string, volumeSizeKB int, err error) {
	cloud, err := v.plugin.getCloudProvider()
	if err != nil {
		return "", 0, err
	}

	volSizeBytes := v.options.Capacity.Value()
	// vSphere works with kilobytes, convert to KiB with rounding up
	volSizeKB := int(volume.RoundUpSize(volSizeBytes, 1024))
	name := volume.GenerateVolumeName(v.options.ClusterName, v.options.PVName, 255)
	vmDiskPath, err = cloud.CreateVolume(name, volSizeKB, v.options.CloudTags)
	if err != nil {
		glog.V(2).Infof("Error creating vsphere volume: %v", err)
		return "", 0, err
	}
	glog.V(2).Infof("Successfully created vsphere volume %s", name)
	return vmDiskPath, volSizeKB, nil
}

// DeleteVolume deletes a vSphere volume.
func (util *VsphereDiskUtil) DeleteVolume(vd *vsphereVolumeDeleter) error {
	cloud, err := vd.plugin.getCloudProvider()
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
