/*
Copyright 2017 The Kubernetes Authors.

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

package rbd

import (
	"fmt"
	"os"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/mount-utils"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
)

// NewAttacher implements AttachableVolumePlugin.NewAttacher.
func (plugin *rbdPlugin) NewAttacher() (volume.Attacher, error) {
	return plugin.newAttacherInternal(&rbdUtil{})
}

// NewDeviceMounter implements DeviceMountableVolumePlugin.NewDeviceMounter
func (plugin *rbdPlugin) NewDeviceMounter() (volume.DeviceMounter, error) {
	return plugin.NewAttacher()
}

func (plugin *rbdPlugin) newAttacherInternal(manager diskManager) (volume.Attacher, error) {
	return &rbdAttacher{
		plugin:  plugin,
		manager: manager,
		mounter: volutil.NewSafeFormatAndMountFromHost(plugin.GetPluginName(), plugin.host),
	}, nil
}

// NewDetacher implements AttachableVolumePlugin.NewDetacher.
func (plugin *rbdPlugin) NewDetacher() (volume.Detacher, error) {
	return plugin.newDetacherInternal(&rbdUtil{})
}

// NewDeviceUnmounter implements DeviceMountableVolumePlugin.NewDeviceUnmounter
func (plugin *rbdPlugin) NewDeviceUnmounter() (volume.DeviceUnmounter, error) {
	return plugin.NewDetacher()
}

func (plugin *rbdPlugin) newDetacherInternal(manager diskManager) (volume.Detacher, error) {
	return &rbdDetacher{
		plugin:  plugin,
		manager: manager,
		mounter: volutil.NewSafeFormatAndMountFromHost(plugin.GetPluginName(), plugin.host),
	}, nil
}

// GetDeviceMountRefs implements AttachableVolumePlugin.GetDeviceMountRefs.
func (plugin *rbdPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter(plugin.GetPluginName())
	return mounter.GetMountRefs(deviceMountPath)
}

func (plugin *rbdPlugin) CanAttach(spec *volume.Spec) (bool, error) {
	return true, nil
}

func (plugin *rbdPlugin) CanDeviceMount(spec *volume.Spec) (bool, error) {
	return true, nil
}

// rbdAttacher implements volume.Attacher interface.
type rbdAttacher struct {
	plugin  *rbdPlugin
	mounter *mount.SafeFormatAndMount
	manager diskManager
}

var _ volume.Attacher = &rbdAttacher{}

var _ volume.DeviceMounter = &rbdAttacher{}

// Attach implements Attacher.Attach.
// We do not lock image here, because it requires kube-controller-manager to
// access external `rbd` utility. And there is no need since AttachDetach
// controller will not try to attach RWO volumes which are already attached to
// other nodes.
func (attacher *rbdAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	return "", nil
}

// VolumesAreAttached implements Attacher.VolumesAreAttached.
// There is no way to confirm whether the volume is attached or not from
// outside of the kubelet node. This method needs to return true always, like
// iSCSI, FC plugin.
func (attacher *rbdAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	volumesAttachedCheck := make(map[*volume.Spec]bool)
	for _, spec := range specs {
		volumesAttachedCheck[spec] = true
	}
	return volumesAttachedCheck, nil
}

// WaitForAttach implements Attacher.WaitForAttach. It's called by kubelet to
// attach volume onto the node.
// This method is idempotent, callers are responsible for retrying on failure.
func (attacher *rbdAttacher) WaitForAttach(spec *volume.Spec, devicePath string, pod *v1.Pod, timeout time.Duration) (string, error) {
	klog.V(4).Infof("rbd: waiting for attach volume (name: %s) for pod (name: %s, uid: %s)", spec.Name(), pod.Name, pod.UID)
	mounter, err := attacher.plugin.createMounterFromVolumeSpecAndPod(spec, pod)
	if err != nil {
		klog.Warningf("failed to create mounter: %v", spec)
		return "", err
	}
	realDevicePath, err := attacher.manager.AttachDisk(*mounter)
	if err != nil {
		return "", err
	}
	klog.V(3).Infof("rbd: successfully wait for attach volume (spec: %s, pool: %s, image: %s) at %s", spec.Name(), mounter.Pool, mounter.Image, realDevicePath)
	return realDevicePath, nil
}

// GetDeviceMountPath implements Attacher.GetDeviceMountPath.
func (attacher *rbdAttacher) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	img, err := getVolumeSourceImage(spec)
	if err != nil {
		return "", err
	}
	pool, err := getVolumeSourcePool(spec)
	if err != nil {
		return "", err
	}
	return makePDNameInternal(attacher.plugin.host, pool, img), nil
}

// MountDevice implements Attacher.MountDevice. It is called by the kubelet to
// mount device at the given mount path.
// This method is idempotent, callers are responsible for retrying on failure.
func (attacher *rbdAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	klog.V(4).Infof("rbd: mouting device %s to %s", devicePath, deviceMountPath)
	notMnt, err := attacher.mounter.IsLikelyNotMountPoint(deviceMountPath)
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
	if !notMnt {
		return nil
	}
	fstype, err := getVolumeSourceFSType(spec)
	if err != nil {
		return err
	}
	ro, err := getVolumeSourceReadOnly(spec)
	if err != nil {
		return err
	}
	options := []string{}
	if ro {
		options = append(options, "ro")
	}
	mountOptions := volutil.MountOptionFromSpec(spec, options...)
	err = attacher.mounter.FormatAndMount(devicePath, deviceMountPath, fstype, mountOptions)
	if err != nil {
		os.Remove(deviceMountPath)
		return fmt.Errorf("rbd: failed to mount device %s at %s (fstype: %s), error %v", devicePath, deviceMountPath, fstype, err)
	}
	klog.V(3).Infof("rbd: successfully mount device %s at %s (fstype: %s)", devicePath, deviceMountPath, fstype)
	return nil
}

// rbdDetacher implements volume.Detacher interface.
type rbdDetacher struct {
	plugin  *rbdPlugin
	manager diskManager
	mounter *mount.SafeFormatAndMount
}

var _ volume.Detacher = &rbdDetacher{}

var _ volume.DeviceUnmounter = &rbdDetacher{}

// UnmountDevice implements Detacher.UnmountDevice. It unmounts the global
// mount of the RBD image. This is called once all bind mounts have been
// unmounted.
// Internally, it does four things:
//  - Unmount device from deviceMountPath.
//  - Detach device from the node.
//  - Remove lock if found. (No need to check volume readonly or not, because
//  device is not on the node anymore, it's safe to remove lock.)
//  - Remove the deviceMountPath at last.
// This method is idempotent, callers are responsible for retrying on failure.
func (detacher *rbdDetacher) UnmountDevice(deviceMountPath string) error {
	if pathExists, pathErr := mount.PathExists(deviceMountPath); pathErr != nil {
		return fmt.Errorf("error checking if path exists: %v", pathErr)
	} else if !pathExists {
		klog.Warningf("Warning: Unmount skipped because path does not exist: %v", deviceMountPath)
		return nil
	}
	devicePath, _, err := mount.GetDeviceNameFromMount(detacher.mounter, deviceMountPath)
	if err != nil {
		return err
	}
	// Unmount the device from the device mount point.
	notMnt, err := detacher.mounter.IsLikelyNotMountPoint(deviceMountPath)
	if err != nil {
		return err
	}
	if !notMnt {
		klog.V(4).Infof("rbd: unmouting device mountpoint %s", deviceMountPath)
		if err = detacher.mounter.Unmount(deviceMountPath); err != nil {
			return err
		}
		klog.V(3).Infof("rbd: successfully umount device mountpath %s", deviceMountPath)
	}

	// Get devicePath from deviceMountPath if devicePath is empty
	if devicePath == "" {
		rbdImageInfo, err := getRbdImageInfo(deviceMountPath)
		if err != nil {
			return err
		}
		found := false
		devicePath, found = getRbdDevFromImageAndPool(rbdImageInfo.pool, rbdImageInfo.name)
		if !found {
			klog.Warningf("rbd: can't found devicePath for %v. Device is already unmounted, Image %v, Pool %v", deviceMountPath, rbdImageInfo.pool, rbdImageInfo.name)
		}
	}

	if devicePath != "" {
		klog.V(4).Infof("rbd: detaching device %s", devicePath)
		err = detacher.manager.DetachDisk(detacher.plugin, deviceMountPath, devicePath)
		if err != nil {
			return err
		}
		klog.V(3).Infof("rbd: successfully detach device %s", devicePath)
	}
	err = os.Remove(deviceMountPath)
	if err != nil {
		return err
	}
	klog.V(3).Infof("rbd: successfully remove device mount point %s", deviceMountPath)
	return nil
}

// Detach implements Detacher.Detach.
func (detacher *rbdDetacher) Detach(volumeName string, nodeName types.NodeName) error {
	return nil
}
