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

package cinder

import (
	"fmt"
	"os"
	"path"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type cinderDiskAttacher struct {
	host           volume.VolumeHost
	cinderProvider CinderProvider
}

var _ volume.Attacher = &cinderDiskAttacher{}

var _ volume.AttachableVolumePlugin = &cinderPlugin{}

const (
	checkSleepDuration = time.Second
)

func (plugin *cinderPlugin) NewAttacher() (volume.Attacher, error) {
	cinder, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}
	return &cinderDiskAttacher{
		host:           plugin.host,
		cinderProvider: cinder,
	}, nil
}

func (plugin *cinderPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter()
	return mount.GetMountRefs(mounter, deviceMountPath)
}

func (attacher *cinderDiskAttacher) Attach(spec *volume.Spec, hostName string) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	volumeID := volumeSource.VolumeID

	instances, res := attacher.cinderProvider.Instances()
	if !res {
		return "", fmt.Errorf("failed to list openstack instances")
	}
	instanceid, err := instances.InstanceID(hostName)
	if err != nil {
		return "", err
	}
	if ind := strings.LastIndex(instanceid, "/"); ind >= 0 {
		instanceid = instanceid[(ind + 1):]
	}
	attached, err := attacher.cinderProvider.DiskIsAttached(volumeID, instanceid)
	if err != nil {
		// Log error and continue with attach
		glog.Warningf(
			"Error checking if volume (%q) is already attached to current node (%q). Will continue and try attach anyway. err=%v",
			volumeID, instanceid, err)
	}

	if err == nil && attached {
		// Volume is already attached to node.
		glog.Infof("Attach operation is successful. volume %q is already attached to node %q.", volumeID, instanceid)
	} else {
		_, err = attacher.cinderProvider.AttachDisk(instanceid, volumeID)
		if err == nil {
			glog.Infof("Attach operation successful: volume %q attached to node %q.", volumeID, instanceid)
		} else {
			glog.Infof("Attach volume %q to instance %q failed with %v", volumeID, instanceid, err)
			return "", err
		}
	}

	devicePath, err := attacher.cinderProvider.GetAttachmentDiskPath(instanceid, volumeID)
	if err != nil {
		glog.Infof("Attach volume %q to instance %q failed with %v", volumeID, instanceid, err)
		return "", err
	}

	return devicePath, err
}

func (attacher *cinderDiskAttacher) WaitForAttach(spec *volume.Spec, devicePath string, timeout time.Duration) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	volumeID := volumeSource.VolumeID

	if devicePath == "" {
		return "", fmt.Errorf("WaitForAttach failed for Cinder disk %q: devicePath is empty.", volumeID)
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
			probeAttachedVolume()
			exists, err := volumeutil.PathExists(devicePath)
			if exists && err == nil {
				glog.Infof("Successfully found attached Cinder disk %q.", volumeID)
				return devicePath, nil
			} else {
				//Log error, if any, and continue checking periodically
				glog.Errorf("Error Stat Cinder disk (%q) is attached: %v", volumeID, err)
			}
		case <-timer.C:
			return "", fmt.Errorf("Could not find attached Cinder disk %q. Timeout waiting for mount paths to be created.", volumeID)
		}
	}
}

func (attacher *cinderDiskAttacher) GetDeviceMountPath(
	spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return makeGlobalPDName(attacher.host, volumeSource.VolumeID), nil
}

// FIXME: this method can be further pruned.
func (attacher *cinderDiskAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	mounter := attacher.host.GetMounter()
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

	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return err
	}

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
	mounter        mount.Interface
	cinderProvider CinderProvider
}

var _ volume.Detacher = &cinderDiskDetacher{}

func (plugin *cinderPlugin) NewDetacher() (volume.Detacher, error) {
	cinder, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}
	return &cinderDiskDetacher{
		mounter:        plugin.host.GetMounter(),
		cinderProvider: cinder,
	}, nil
}

func (detacher *cinderDiskDetacher) Detach(deviceMountPath string, hostName string) error {
	volumeID := path.Base(deviceMountPath)
	instances, res := detacher.cinderProvider.Instances()
	if !res {
		return fmt.Errorf("failed to list openstack instances")
	}
	instanceid, err := instances.InstanceID(hostName)
	if ind := strings.LastIndex(instanceid, "/"); ind >= 0 {
		instanceid = instanceid[(ind + 1):]
	}

	attached, err := detacher.cinderProvider.DiskIsAttached(volumeID, instanceid)
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

	if err = detacher.cinderProvider.DetachDisk(instanceid, volumeID); err != nil {
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
			if pathExists, err := volumeutil.PathExists(devicePath); err != nil {
				return fmt.Errorf("Error checking if device path exists: %v", err)
			} else if !pathExists {
				return nil
			}
		case <-timer.C:
			return fmt.Errorf("Timeout reached; PD Device %v is still attached", devicePath)
		}
	}
}

func (detacher *cinderDiskDetacher) UnmountDevice(deviceMountPath string) error {
	return volumeutil.UnmountPath(deviceMountPath, detacher.mounter)
}
