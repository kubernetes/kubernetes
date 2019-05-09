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
	"context"
	"fmt"
	"os"
	"path"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type cinderDiskAttacher struct {
	host           volume.VolumeHost
	cinderProvider BlockStorageProvider
}

var _ volume.Attacher = &cinderDiskAttacher{}

var _ volume.DeviceMounter = &cinderDiskAttacher{}

var _ volume.AttachableVolumePlugin = &cinderPlugin{}

var _ volume.DeviceMountableVolumePlugin = &cinderPlugin{}

const (
	probeVolumeInitDelay     = 1 * time.Second
	probeVolumeFactor        = 2.0
	operationFinishInitDelay = 1 * time.Second
	operationFinishFactor    = 1.1
	operationFinishSteps     = 10
	diskAttachInitDelay      = 1 * time.Second
	diskAttachFactor         = 1.2
	diskAttachSteps          = 15
	diskDetachInitDelay      = 1 * time.Second
	diskDetachFactor         = 1.2
	diskDetachSteps          = 13
)

func (plugin *cinderPlugin) NewAttacher() (volume.Attacher, error) {
	cinder, err := plugin.getCloudProvider()
	if err != nil {
		return nil, err
	}
	return &cinderDiskAttacher{
		host:           plugin.host,
		cinderProvider: cinder,
	}, nil
}

func (plugin *cinderPlugin) NewDeviceMounter() (volume.DeviceMounter, error) {
	return plugin.NewAttacher()
}

func (plugin *cinderPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter(plugin.GetPluginName())
	return mounter.GetMountRefs(deviceMountPath)
}

func (attacher *cinderDiskAttacher) waitOperationFinished(volumeID string) error {
	backoff := wait.Backoff{
		Duration: operationFinishInitDelay,
		Factor:   operationFinishFactor,
		Steps:    operationFinishSteps,
	}

	var volumeStatus string
	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		var pending bool
		var err error
		pending, volumeStatus, err = attacher.cinderProvider.OperationPending(volumeID)
		if err != nil {
			return false, err
		}
		return !pending, nil
	})

	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("Volume %q is %s, can't finish within the alloted time", volumeID, volumeStatus)
	}

	return err
}

func (attacher *cinderDiskAttacher) waitDiskAttached(instanceID, volumeID string) error {
	backoff := wait.Backoff{
		Duration: diskAttachInitDelay,
		Factor:   diskAttachFactor,
		Steps:    diskAttachSteps,
	}

	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		attached, err := attacher.cinderProvider.DiskIsAttached(instanceID, volumeID)
		if err != nil {
			return false, err
		}
		return attached, nil
	})

	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("Volume %q failed to be attached within the alloted time", volumeID)
	}

	return err
}

func (attacher *cinderDiskAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	volumeID, _, _, err := getVolumeInfo(spec)
	if err != nil {
		return "", err
	}

	instanceID, err := attacher.nodeInstanceID(nodeName)
	if err != nil {
		return "", err
	}

	if err := attacher.waitOperationFinished(volumeID); err != nil {
		return "", err
	}

	attached, err := attacher.cinderProvider.DiskIsAttached(instanceID, volumeID)
	if err != nil {
		// Log error and continue with attach
		klog.Warningf(
			"Error checking if volume (%q) is already attached to current instance (%q). Will continue and try attach anyway. err=%v",
			volumeID, instanceID, err)
	}

	if err == nil && attached {
		// Volume is already attached to instance.
		klog.Infof("Attach operation is successful. volume %q is already attached to instance %q.", volumeID, instanceID)
	} else {
		_, err = attacher.cinderProvider.AttachDisk(instanceID, volumeID)
		if err == nil {
			if err = attacher.waitDiskAttached(instanceID, volumeID); err != nil {
				klog.Errorf("Error waiting for volume %q to be attached from node %q: %v", volumeID, nodeName, err)
				return "", err
			}
			klog.Infof("Attach operation successful: volume %q attached to instance %q.", volumeID, instanceID)
		} else {
			klog.Infof("Attach volume %q to instance %q failed with: %v", volumeID, instanceID, err)
			return "", err
		}
	}

	devicePath, err := attacher.cinderProvider.GetAttachmentDiskPath(instanceID, volumeID)
	if err != nil {
		klog.Infof("Can not get device path of volume %q which be attached to instance %q, failed with: %v", volumeID, instanceID, err)
		return "", err
	}

	return devicePath, nil
}

func (attacher *cinderDiskAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	volumesAttachedCheck := make(map[*volume.Spec]bool)
	volumeSpecMap := make(map[string]*volume.Spec)
	volumeIDList := []string{}
	for _, spec := range specs {
		volumeID, _, _, err := getVolumeInfo(spec)
		if err != nil {
			klog.Errorf("Error getting volume (%q) source : %v", spec.Name(), err)
			continue
		}

		volumeIDList = append(volumeIDList, volumeID)
		volumesAttachedCheck[spec] = true
		volumeSpecMap[volumeID] = spec
	}

	attachedResult, err := attacher.cinderProvider.DisksAreAttachedByName(nodeName, volumeIDList)
	if err != nil {
		// Log error and continue with attach
		klog.Errorf(
			"Error checking if Volumes (%v) are already attached to current node (%q). Will continue and try attach anyway. err=%v",
			volumeIDList, nodeName, err)
		return volumesAttachedCheck, err
	}

	for volumeID, attached := range attachedResult {
		if !attached {
			spec := volumeSpecMap[volumeID]
			volumesAttachedCheck[spec] = false
			klog.V(2).Infof("VolumesAreAttached: check volume %q (specName: %q) is no longer attached", volumeID, spec.Name())
		}
	}
	return volumesAttachedCheck, nil
}

func (attacher *cinderDiskAttacher) WaitForAttach(spec *volume.Spec, devicePath string, _ *v1.Pod, timeout time.Duration) (string, error) {
	// NOTE: devicePath is path as reported by Cinder, which may be incorrect and should not be used. See Issue #33128
	volumeID, _, _, err := getVolumeInfo(spec)
	if err != nil {
		return "", err
	}

	if devicePath == "" {
		return "", fmt.Errorf("WaitForAttach failed for Cinder disk %q: devicePath is empty", volumeID)
	}

	ticker := time.NewTicker(probeVolumeInitDelay)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	duration := probeVolumeInitDelay
	for {
		select {
		case <-ticker.C:
			klog.V(5).Infof("Checking Cinder disk %q is attached.", volumeID)
			probeAttachedVolume()
			if !attacher.cinderProvider.ShouldTrustDevicePath() {
				// Using the Cinder volume ID, find the real device path (See Issue #33128)
				devicePath = attacher.cinderProvider.GetDevicePath(volumeID)
			}
			exists, err := mount.PathExists(devicePath)
			if exists && err == nil {
				klog.Infof("Successfully found attached Cinder disk %q at %v.", volumeID, devicePath)
				return devicePath, nil
			}
			// Log an error, and continue checking periodically
			klog.Errorf("Error: could not find attached Cinder disk %q (path: %q): %v", volumeID, devicePath, err)
			// Using exponential backoff instead of linear
			ticker.Stop()
			duration = time.Duration(float64(duration) * probeVolumeFactor)
			ticker = time.NewTicker(duration)
		case <-timer.C:
			return "", fmt.Errorf("could not find attached Cinder disk %q. Timeout waiting for mount paths to be created", volumeID)
		}
	}
}

func (attacher *cinderDiskAttacher) GetDeviceMountPath(
	spec *volume.Spec) (string, error) {
	volumeID, _, _, err := getVolumeInfo(spec)
	if err != nil {
		return "", err
	}

	return makeGlobalPDName(attacher.host, volumeID), nil
}

// FIXME: this method can be further pruned.
func (attacher *cinderDiskAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	mounter := attacher.host.GetMounter(cinderVolumePluginName)
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

	_, volumeFSType, readOnly, err := getVolumeInfo(spec)
	if err != nil {
		return err
	}

	options := []string{}
	if readOnly {
		options = append(options, "ro")
	}
	if notMnt {
		diskMounter := volumeutil.NewSafeFormatAndMountFromHost(cinderVolumePluginName, attacher.host)
		mountOptions := volumeutil.MountOptionFromSpec(spec, options...)
		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, volumeFSType, mountOptions)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
	}
	return nil
}

type cinderDiskDetacher struct {
	mounter        mount.Interface
	cinderProvider BlockStorageProvider
}

var _ volume.Detacher = &cinderDiskDetacher{}

var _ volume.DeviceUnmounter = &cinderDiskDetacher{}

func (plugin *cinderPlugin) NewDetacher() (volume.Detacher, error) {
	cinder, err := plugin.getCloudProvider()
	if err != nil {
		return nil, err
	}
	return &cinderDiskDetacher{
		mounter:        plugin.host.GetMounter(plugin.GetPluginName()),
		cinderProvider: cinder,
	}, nil
}

func (plugin *cinderPlugin) NewDeviceUnmounter() (volume.DeviceUnmounter, error) {
	return plugin.NewDetacher()
}

func (detacher *cinderDiskDetacher) waitOperationFinished(volumeID string) error {
	backoff := wait.Backoff{
		Duration: operationFinishInitDelay,
		Factor:   operationFinishFactor,
		Steps:    operationFinishSteps,
	}

	var volumeStatus string
	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		var pending bool
		var err error
		pending, volumeStatus, err = detacher.cinderProvider.OperationPending(volumeID)
		if err != nil {
			return false, err
		}
		return !pending, nil
	})

	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("Volume %q is %s, can't finish within the alloted time", volumeID, volumeStatus)
	}

	return err
}

func (detacher *cinderDiskDetacher) waitDiskDetached(instanceID, volumeID string) error {
	backoff := wait.Backoff{
		Duration: diskDetachInitDelay,
		Factor:   diskDetachFactor,
		Steps:    diskDetachSteps,
	}

	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		attached, err := detacher.cinderProvider.DiskIsAttached(instanceID, volumeID)
		if err != nil {
			return false, err
		}
		return !attached, nil
	})

	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("Volume %q failed to detach within the alloted time", volumeID)
	}

	return err
}

func (detacher *cinderDiskDetacher) Detach(volumeName string, nodeName types.NodeName) error {
	volumeID := path.Base(volumeName)
	if err := detacher.waitOperationFinished(volumeID); err != nil {
		return err
	}
	attached, instanceID, err := detacher.cinderProvider.DiskIsAttachedByName(nodeName, volumeID)
	if err != nil {
		// Log error and continue with detach
		klog.Errorf(
			"Error checking if volume (%q) is already attached to current node (%q). Will continue and try detach anyway. err=%v",
			volumeID, nodeName, err)
	}

	if err == nil && !attached {
		// Volume is already detached from node.
		klog.Infof("detach operation was successful. volume %q is already detached from node %q.", volumeID, nodeName)
		return nil
	}

	if err = detacher.cinderProvider.DetachDisk(instanceID, volumeID); err != nil {
		klog.Errorf("Error detaching volume %q from node %q: %v", volumeID, nodeName, err)
		return err
	}
	if err = detacher.waitDiskDetached(instanceID, volumeID); err != nil {
		klog.Errorf("Error waiting for volume %q to detach from node %q: %v", volumeID, nodeName, err)
		return err
	}
	klog.Infof("detached volume %q from node %q", volumeID, nodeName)
	return nil
}

func (detacher *cinderDiskDetacher) UnmountDevice(deviceMountPath string) error {
	return mount.CleanupMountPoint(deviceMountPath, detacher.mounter, false)
}

func (plugin *cinderPlugin) CanAttach(spec *volume.Spec) (bool, error) {
	return true, nil
}

func (plugin *cinderPlugin) CanDeviceMount(spec *volume.Spec) (bool, error) {
	return true, nil
}

func (attacher *cinderDiskAttacher) nodeInstanceID(nodeName types.NodeName) (string, error) {
	instances, res := attacher.cinderProvider.Instances()
	if !res {
		return "", fmt.Errorf("failed to list openstack instances")
	}
	instanceID, err := instances.InstanceID(context.TODO(), nodeName)
	if err != nil {
		return "", err
	}
	if ind := strings.LastIndex(instanceID, "/"); ind >= 0 {
		instanceID = instanceID[(ind + 1):]
	}
	return instanceID, nil
}
