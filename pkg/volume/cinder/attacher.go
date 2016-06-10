/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package cinder

import (
	"fmt"
	"os"
	"path"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

type cinderDiskAttacher struct {
	host volume.VolumeHost
}

var _ volume.Attacher = &cinderDiskAttacher{}

var _ volume.AttachableVolumePlugin = &cinderPlugin{}

const (
	checkSleepDuration = time.Second
)

func (plugin *cinderPlugin) NewAttacher() (volume.Attacher, error) {
	return &cinderDiskAttacher{host: plugin.host}, nil
}

func (plugin *cinderPlugin) GetDeviceName(spec *volume.Spec) (string, error) {
	volumeSource, _ := getVolumeSource(spec)
	if volumeSource == nil {
		return "", fmt.Errorf("Spec does not reference a Cinder volume type")
	}

	return volumeSource.VolumeID, nil
}

func (attacher *cinderDiskAttacher) Attach(spec *volume.Spec, hostName string) error {
	volumeSource, _ := getVolumeSource(spec)
	volumeID := volumeSource.VolumeID

	cloud, err := getCloudProvider(attacher.host.GetCloudProvider())
	if err != nil {
		return err
	}
	instances, res := cloud.Instances()
	if !res {
		return fmt.Errorf("failed to list openstack instances")
	}
	instanceid, err := instances.InstanceID(hostName)
	if err != nil {
		return err
	}
	if ind := strings.LastIndex(instanceid, "/"); ind >= 0 {
		instanceid = instanceid[(ind + 1):]
	}
	attached, err := cloud.DiskIsAttached(volumeID, instanceid)
	if err != nil {
		// Log error and continue with attach
		glog.Errorf(
			"Error checking if volume (%q) is already attached to current node (%q). Will continue and try attach anyway. err=%v",
			volumeID, instanceid, err)
	}

	if err == nil && attached {
		// Volume is already attached to node.
		glog.Infof("Attach operation is successful. volume %q is already attached to node %q.", volumeID, instanceid)
		return nil
	}

	_, err = cloud.AttachDisk(instanceid, volumeID)
	if err != nil {
		glog.Infof("attach volume %q to instance %q gets %v", volumeID, instanceid, err)
	}
	glog.Infof("attached volume %q to instance %q", volumeID, instanceid)
	return err
}

func (attacher *cinderDiskAttacher) WaitForAttach(spec *volume.Spec, timeout time.Duration) (string, error) {
	cloud, err := getCloudProvider(attacher.host.GetCloudProvider())
	if err != nil {
		return "", err
	}
	volumeSource, _ := getVolumeSource(spec)
	volumeID := volumeSource.VolumeID
	instanceid, err := cloud.InstanceID()
	if err != nil {
		return "", err
	}
	devicePath := ""
	if d, err := cloud.GetAttachmentDiskPath(instanceid, volumeID); err == nil {
		devicePath = d
	} else {
		glog.Errorf("%q GetAttachmentDiskPath (%q) gets error %v", instanceid, volumeID, err)
	}

	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		probeAttachedVolume()
		select {
		case <-ticker.C:
			glog.V(5).Infof("Checking Cinder disk %q is attached.", volumeID)
			if devicePath == "" {
				if d, err := cloud.GetAttachmentDiskPath(instanceid, volumeID); err == nil {
					devicePath = d
				} else {
					glog.Errorf("%q GetAttachmentDiskPath (%q) gets error %v", instanceid, volumeID, err)
				}
			}
			if devicePath == "" {
				glog.V(5).Infof("Cinder disk (%q) is not attached yet", volumeID)
			} else {
				probeAttachedVolume()
				exists, err := pathExists(devicePath)
				if exists && err == nil {
					glog.Infof("Successfully found attached Cinder disk %q.", volumeID)
					return devicePath, nil
				} else {
					//Log error, if any, and continue checking periodically
					glog.Errorf("Error Stat Cinder disk (%q) is attached: %v", volumeID, err)
				}
			}
		case <-timer.C:
			return "", fmt.Errorf("Could not find attached Cinder disk %q. Timeout waiting for mount paths to be created.", volumeID)
		}
	}
}

func (attacher *cinderDiskAttacher) GetDeviceMountPath(spec *volume.Spec) string {
	volumeSource, _ := getVolumeSource(spec)
	return makeGlobalPDName(attacher.host, volumeSource.VolumeID)
}

// FIXME: this method can be further pruned.
func (attacher *cinderDiskAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string, mounter mount.Interface) error {
	notMnt, err := mounter.IsLikelyNotMountPoint(deviceMountPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(deviceMountPath, 0750); err != nil {
				return err
			}
			notMnt = true
		} else {
			return err
		}
	}

	volumeSource, readOnly := getVolumeSource(spec)

	options := []string{}
	if readOnly {
		options = append(options, "ro")
	}
	if notMnt {
		diskMounter := &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}
		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, volumeSource.FSType, options)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
	}
	return nil
}

type cinderDiskDetacher struct {
	host volume.VolumeHost
}

var _ volume.Detacher = &cinderDiskDetacher{}

func (plugin *cinderPlugin) NewDetacher() (volume.Detacher, error) {
	return &cinderDiskDetacher{host: plugin.host}, nil
}

func (detacher *cinderDiskDetacher) Detach(deviceMountPath string, hostName string) error {
	volumeID := path.Base(deviceMountPath)
	cloud, err := getCloudProvider(detacher.host.GetCloudProvider())
	if err != nil {
		return err
	}
	instances, res := cloud.Instances()
	if !res {
		return fmt.Errorf("failed to list openstack instances")
	}
	instanceid, err := instances.InstanceID(hostName)
	if ind := strings.LastIndex(instanceid, "/"); ind >= 0 {
		instanceid = instanceid[(ind + 1):]
	}

	attached, err := cloud.DiskIsAttached(volumeID, instanceid)
	if err != nil {
		// Log error and continue with detach
		glog.Errorf(
			"Error checking if volume (%q) is already attached to current node (%q). Will continue and try detach anyway. err=%v",
			volumeID, hostName, err)
	}

	if err == nil && !attached {
		// Volume is already detached from node.
		glog.Infof("detach operation was successful. volume %q is already detached from node %q.", volumeID, hostName)
		return nil
	}

	if err = cloud.DetachDisk(instanceid, volumeID); err != nil {
		glog.Errorf("Error detaching volume %q: %v", volumeID, err)
		return err
	}
	glog.Infof("detatached volume %q from instance %q", volumeID, instanceid)
	return nil
}

func (detacher *cinderDiskDetacher) WaitForDetach(devicePath string, timeout time.Duration) error {
	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof("Checking device %q is detached.", devicePath)
			if pathExists, err := pathExists(devicePath); err != nil {
				return fmt.Errorf("Error checking if device path exists: %v", err)
			} else if !pathExists {
				return nil
			}
		case <-timer.C:
			return fmt.Errorf("Timeout reached; PD Device %v is still attached", devicePath)
		}
	}
}

func (detacher *cinderDiskDetacher) UnmountDevice(deviceMountPath string, mounter mount.Interface) error {
	volume := path.Base(deviceMountPath)
	if err := unmountPDAndRemoveGlobalPath(deviceMountPath, mounter); err != nil {
		glog.Errorf("Error unmounting %q: %v", volume, err)
	}

	return nil
}

func getVolumeSource(spec *volume.Spec) (*api.CinderVolumeSource, bool) {
	var readOnly bool
	var volumeSource *api.CinderVolumeSource

	if spec.Volume != nil && spec.Volume.Cinder != nil {
		volumeSource = spec.Volume.Cinder
		readOnly = volumeSource.ReadOnly
	} else {
		volumeSource = spec.PersistentVolume.Spec.Cinder
		readOnly = spec.ReadOnly
	}

	return volumeSource, readOnly
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

// Unmount the global mount path, which should be the only one, and delete it.
func unmountPDAndRemoveGlobalPath(globalMountPath string, mounter mount.Interface) error {
	err := mounter.Unmount(globalMountPath)
	os.Remove(globalMountPath)
	return err
}
