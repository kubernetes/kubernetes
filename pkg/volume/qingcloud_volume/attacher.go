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

package qingcloud_volume

import (
	"fmt"
	"os"
	"path"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/qingcloud"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type qingcloudVolumeAttacher struct {
	host      volume.VolumeHost
	qcVolumes qingcloud.Volumes
}

var _ volume.Attacher = &qingcloudVolumeAttacher{}

var _ volume.AttachableVolumePlugin = &qingcloudVolumePlugin{}

func (plugin *qingcloudVolumePlugin) NewAttacher() (volume.Attacher, error) {
	qingCloud, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}

	return &qingcloudVolumeAttacher{
		host:      plugin.host,
		qcVolumes: qingCloud,
	}, nil
}

func (plugin *qingcloudVolumePlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter()
	return mount.GetMountRefs(mounter, deviceMountPath)
}

func (attacher *qingcloudVolumeAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	volumeID := volumeSource.VolumeID

	// qingcloud.AttachVolume checks if disk is already attached to node and
	// succeeds in that case, so no need to do that separately.
	devicePath, err := attacher.qcVolumes.AttachVolume(volumeID, nodeName)
	if err != nil {
		glog.Errorf("Error attaching volume %q: %+v", volumeID, err)
		return "", err
	}

	return devicePath, nil
}

func (attacher *qingcloudVolumeAttacher) WaitForAttach(spec *volume.Spec, devicePath string, timeout time.Duration) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	volumeID := volumeSource.VolumeID

	if devicePath == "" {
		return "", fmt.Errorf("WaitForAttach failed for qingcloud Volume %q: devicePath is empty.", volumeID)
	}

	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof("Checking qingcloud volume %q is attached.", volumeID)
			exists, err := volumeutil.PathExists(devicePath)
			if err != nil {
				// Log error, if any, and continue checking periodically.
				glog.Errorf("Error verifying qingcloud volume (%q) is attached: %v", volumeID, err)
			} else if exists {
				// A device path has successfully been created for the PD
				glog.Infof("Successfully found attached qingcloud volume %q.", volumeID)
				return devicePath, nil
			}
		case <-timer.C:
			return "", fmt.Errorf("Could not find attached qingcloud volume %q. Timeout waiting for mount paths to be created.", volumeID)
		}
	}
}

func (attacher *qingcloudVolumeAttacher) GetDeviceMountPath(
	spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return makeGlobalPDPath(attacher.host, volumeSource.VolumeID), nil
}

func (attacher *qingcloudVolumeAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
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
		volumeMounter := &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}
		err = volumeMounter.FormatAndMount(devicePath, deviceMountPath, volumeSource.FSType, options)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
	}
	return nil
}

type qingcloudVolumeDetacher struct {
	mounter     mount.Interface
	qingVolumes qingcloud.Volumes
}

var _ volume.Detacher = &qingcloudVolumeDetacher{}

func (plugin *qingcloudVolumePlugin) NewDetacher() (volume.Detacher, error) {
	qingCloud, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}

	return &qingcloudVolumeDetacher{
		mounter:     plugin.host.GetMounter(),
		qingVolumes: qingCloud,
	}, nil
}

func (detacher *qingcloudVolumeDetacher) Detach(deviceMountPath string, nodeName types.NodeName) error {
	volumeID := path.Base(deviceMountPath)

	attached, err := detacher.qingVolumes.VolumeIsAttached(volumeID, nodeName)
	if err != nil {
		// Log error and continue with detach
		glog.Errorf(
			"Error checking if volume (%q) is already attached to current node (%v). Will continue and try detach anyway. err=%v",
			volumeID, nodeName, err)
	}

	if err == nil && !attached {
		// Volume is already detached from node.
		glog.Infof("detach operation was successful. volume %q is already detached from node %v.", volumeID, nodeName)
		return nil
	}

	if err = detacher.qingVolumes.DetachVolume(volumeID, nodeName); err != nil {
		glog.Errorf("Error detaching volumeID %q: %v", volumeID, err)
		return err
	}
	return nil
}

func (detacher *qingcloudVolumeDetacher) WaitForDetach(devicePath string, timeout time.Duration) error {
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

func (detacher *qingcloudVolumeDetacher) UnmountDevice(deviceMountPath string) error {
	return volumeutil.UnmountPath(deviceMountPath, detacher.mounter)
}
