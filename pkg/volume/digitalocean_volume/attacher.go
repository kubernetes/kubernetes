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

package digitalocean_volume

import (
	"fmt"
	"os"
	"path"
	"strconv"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type doVolumeAttacher struct {
	host       volume.VolumeHost
	doProvider DoProvider
}

var _ volume.Attacher = &doVolumeAttacher{}

var _ volume.AttachableVolumePlugin = &doVolumePlugin{}

const (
	checkSleepDuration = time.Second
)

func (plugin *doVolumePlugin) NewAttacher() (volume.Attacher, error) {
	doProvider, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}
	return &doVolumeAttacher{
		host:       plugin.host,
		doProvider: doProvider,
	}, nil
}

func (plugin *doVolumePlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter()
	return mount.GetMountRefs(mounter, deviceMountPath)
}

func (attacher *doVolumeAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	volumeID := volumeSource.VolumeID

	instances, res := attacher.doProvider.Instances()
	if !res {
		return "", fmt.Errorf("failed to list digitalocean instances")
	}
	instanceID, err := instances.InstanceID(nodeName)
	if err != nil {
		return "", err
	}
	intInstanceID, _ := strconv.Atoi(instanceID)
	attached, err := attacher.doProvider.VolumeIsAttached(volumeID, intInstanceID)
	if err != nil {
		// Log error and continue with attach
		glog.Warningf(
			"Error checking if volume (%q) is already attached to current node (%q). Will continue and try attach anyway. err=%v",
			volumeID, instanceID, err)
	}

	if err == nil && attached {
		// Volume is already attached to node.
		glog.Infof("Attach operation is successful. volume %q is already attached to node %q.", volumeID, instanceID)
	} else {
		_, err = attacher.doProvider.AttachVolume(intInstanceID, volumeID)
		if err == nil {
			glog.Infof("Attach operation successful: volume %q attached to node %q.", volumeID, instanceID)
		} else {
			glog.Infof("Attach volume %q to instance %q failed with %v", volumeID, instanceID, err)
			return "", err
		}
	}

	devicePath, err := attacher.doProvider.GetAttachmentVolumePath(intInstanceID, volumeID)
	if err != nil {
		glog.Infof("Attach volume %q to instance %q failed with %v", volumeID, instanceID, err)
		return "", err
	}

	return devicePath, err
}

func (attacher *doVolumeAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	volumesAttachedCheck := make(map[*volume.Spec]bool)
	volumeSpecMap := make(map[string]*volume.Spec)
	volumeIDList := []string{}
	for _, spec := range specs {
		volumeSource, _, err := getVolumeSource(spec)
		if err != nil {
			glog.Errorf("Error getting volume (%q) source : %v", spec.Name(), err)
			continue
		}

		volumeIDList = append(volumeIDList, volumeSource.VolumeID)
		volumesAttachedCheck[spec] = true
		volumeSpecMap[volumeSource.VolumeID] = spec
	}
	instances, res := attacher.doProvider.Instances()
	if !res {
		return volumesAttachedCheck, fmt.Errorf("failed to list digitalocean instances")
	}
	instanceID, err := instances.InstanceID(nodeName)
	if err != nil {
		return volumesAttachedCheck, err
	}
	intInstanceID, _ := strconv.Atoi(instanceID)
	attachedResult, err := attacher.doProvider.VolumesAreAttached(volumeIDList, intInstanceID)
	if err != nil {
		// Log error and continue with attach
		glog.Errorf(
			"Error checking if Volumes (%v) are already attached to current node (%q). Will continue and try attach anyway. err=%v",
			volumeIDList, nodeName, err)
		return volumesAttachedCheck, err
	}

	for volumeID, attached := range attachedResult {
		if !attached {
			spec := volumeSpecMap[volumeID]
			volumesAttachedCheck[spec] = false
			glog.V(2).Infof("VolumesAreAttached: check volume %q (specName: %q) is no longer attached", volumeID, spec.Name())
		}
	}
	return volumesAttachedCheck, nil
}

func (attacher *doVolumeAttacher) WaitForAttach(spec *volume.Spec, devicePath string, timeout time.Duration) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	volumeID := volumeSource.VolumeID

	if devicePath == "" {
		return "", fmt.Errorf("WaitForAttach failed for DigitalOcean Volume %q: devicePath is empty.", volumeID)
	}

	err = wait.Poll(checkSleepDuration, timeout, func() (bool, error) {
		glog.V(4).Infof("Checking DigitalOcean Volume %q (device path %s) is attached.", volumeID, devicePath)
		probeAttachedVolume()
		exists, err := volumeutil.PathExists(devicePath)
		if exists && err == nil {
			glog.V(4).Infof("Successfully found attached DigitalOcean Volume %q (device path %s).", volumeID, devicePath)
			return true, nil
		} else {
			//Log error, if any, and continue checking periodically
			glog.V(4).Infof("Error Stat DigitalOcean volume (%q) is attached: %v", volumeID, err)
			return false, nil
		}
	})

	return devicePath, err
}

func (attacher *doVolumeAttacher) GetDeviceMountPath(
	spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}
	return makeGlobalPDName(attacher.host, volumeSource.VolumeID), nil
}

func (attacher *doVolumeAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
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

type doVolumeDetacher struct {
	mounter    mount.Interface
	doProvider DoProvider
}

var _ volume.Detacher = &doVolumeDetacher{}

func (plugin *doVolumePlugin) NewDetacher() (volume.Detacher, error) {
	doProvider, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}
	return &doVolumeDetacher{
		mounter:    plugin.host.GetMounter(),
		doProvider: doProvider,
	}, nil
}

func (detacher *doVolumeDetacher) Detach(deviceMountPath string, nodeName types.NodeName) error {
	volumeID := path.Base(deviceMountPath)
	instances, res := detacher.doProvider.Instances()
	if !res {
		return fmt.Errorf("failed to list digitalocean instances")
	}
	instanceID, err := instances.InstanceID(nodeName)
	intInstanceID, _ := strconv.Atoi(instanceID)

	attached, err := detacher.doProvider.VolumeIsAttached(volumeID, intInstanceID)
	if err != nil {
		// Log error and continue with detach
		glog.Errorf(
			"Error checking if volume (%q) is already attached to current node (%q). Will continue and try detach anyway. err=%v",
			volumeID, nodeName, err)
	}

	if err == nil && !attached {
		// Volume is already detached from node.
		glog.Infof("detach operation was successful. volume %q is already detached from node %q.", volumeID, nodeName)
		return nil
	}

	if err = detacher.doProvider.DetachVolume(intInstanceID, volumeID); err != nil {
		glog.Errorf("Error detaching volume %q: %v", volumeID, err)
		return err
	}
	glog.Infof("detatached volume %q from instance %q", volumeID, instanceID)
	return nil
}

func (detacher *doVolumeDetacher) WaitForDetach(devicePath string, timeout time.Duration) error {
	return wait.Poll(checkSleepDuration, timeout, func() (bool, error) {
		glog.V(4).Infof("Checking device %q is detached.", devicePath)
		if pathExists, err := volumeutil.PathExists(devicePath); err != nil {
			return false, fmt.Errorf("Error checking if device path exists: %v", err)
		} else if !pathExists {
			return true, nil
		} else {
			return false, nil
		}
	})
}

func (detacher *doVolumeDetacher) UnmountDevice(deviceMountPath string) error {
	return volumeutil.UnmountPath(deviceMountPath, detacher.mounter)
}
