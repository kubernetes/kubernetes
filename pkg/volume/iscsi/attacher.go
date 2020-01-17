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

package iscsi

import (
	"fmt"
	"os"
	"time"

	"k8s.io/klog"
	"k8s.io/utils/keymutex"
	"k8s.io/utils/mount"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type iscsiAttacher struct {
	host        volume.VolumeHost
	targetLocks keymutex.KeyMutex
	manager     diskManager
}

var _ volume.Attacher = &iscsiAttacher{}

var _ volume.DeviceMounter = &iscsiAttacher{}

var _ volume.AttachableVolumePlugin = &iscsiPlugin{}

var _ volume.DeviceMountableVolumePlugin = &iscsiPlugin{}

func (plugin *iscsiPlugin) NewAttacher() (volume.Attacher, error) {
	return &iscsiAttacher{
		host:        plugin.host,
		targetLocks: plugin.targetLocks,
		manager:     &ISCSIUtil{},
	}, nil
}

func (plugin *iscsiPlugin) NewDeviceMounter() (volume.DeviceMounter, error) {
	return plugin.NewAttacher()
}

func (plugin *iscsiPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter(iscsiPluginName)
	return mounter.GetMountRefs(deviceMountPath)
}

func (attacher *iscsiAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	return "", nil
}

func (attacher *iscsiAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	volumesAttachedCheck := make(map[*volume.Spec]bool)
	for _, spec := range specs {
		volumesAttachedCheck[spec] = true
	}

	return volumesAttachedCheck, nil
}

func (attacher *iscsiAttacher) WaitForAttach(spec *volume.Spec, devicePath string, pod *v1.Pod, timeout time.Duration) (string, error) {
	mounter, err := volumeSpecToMounter(spec, attacher.host, attacher.targetLocks, pod)
	if err != nil {
		klog.Warningf("failed to get iscsi mounter: %v", err)
		return "", err
	}
	return attacher.manager.AttachDisk(*mounter)
}

func (attacher *iscsiAttacher) GetDeviceMountPath(
	spec *volume.Spec) (string, error) {
	mounter, err := volumeSpecToMounter(spec, attacher.host, attacher.targetLocks, nil)
	if err != nil {
		klog.Warningf("failed to get iscsi mounter: %v", err)
		return "", err
	}
	if mounter.InitiatorName != "" {
		// new iface name is <target portal>:<volume name>
		mounter.Iface = mounter.Portals[0] + ":" + mounter.VolName
	}
	return attacher.manager.MakeGlobalPDName(*mounter.iscsiDisk), nil
}

func (attacher *iscsiAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	mounter := attacher.host.GetMounter(iscsiPluginName)
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
	readOnly, fsType, err := getISCSIVolumeInfo(spec)
	if err != nil {
		return err
	}

	options := []string{}
	if readOnly {
		options = append(options, "ro")
	}
	if notMnt {
		diskMounter := &mount.SafeFormatAndMount{Interface: mounter, Exec: attacher.host.GetExec(iscsiPluginName)}
		mountOptions := volumeutil.MountOptionFromSpec(spec, options...)
		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, fsType, mountOptions)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
	}
	return nil
}

type iscsiDetacher struct {
	host    volume.VolumeHost
	mounter mount.Interface
	manager diskManager
	plugin  *iscsiPlugin
}

var _ volume.Detacher = &iscsiDetacher{}

var _ volume.DeviceUnmounter = &iscsiDetacher{}

func (plugin *iscsiPlugin) NewDetacher() (volume.Detacher, error) {
	return &iscsiDetacher{
		host:    plugin.host,
		mounter: plugin.host.GetMounter(iscsiPluginName),
		manager: &ISCSIUtil{},
		plugin:  plugin,
	}, nil
}

func (plugin *iscsiPlugin) NewDeviceUnmounter() (volume.DeviceUnmounter, error) {
	return plugin.NewDetacher()
}

func (detacher *iscsiDetacher) Detach(volumeName string, nodeName types.NodeName) error {
	return nil
}

func (detacher *iscsiDetacher) UnmountDevice(deviceMountPath string) error {
	unMounter := volumeSpecToUnmounter(detacher.mounter, detacher.host, detacher.plugin)
	err := detacher.manager.DetachDisk(*unMounter, deviceMountPath)
	if err != nil {
		return fmt.Errorf("iscsi: failed to detach disk: %s\nError: %v", deviceMountPath, err)
	}
	klog.V(4).Infof("iscsi: %q is unmounted, deleting the directory", deviceMountPath)
	err = os.RemoveAll(deviceMountPath)
	if err != nil {
		return fmt.Errorf("iscsi: failed to delete the directory: %s\nError: %v", deviceMountPath, err)
	}
	klog.V(4).Infof("iscsi: successfully detached disk: %s", deviceMountPath)
	return nil
}

func (plugin *iscsiPlugin) CanAttach(spec *volume.Spec) (bool, error) {
	return true, nil
}

func (plugin *iscsiPlugin) CanDeviceMount(spec *volume.Spec) (bool, error) {
	return true, nil
}

func volumeSpecToMounter(spec *volume.Spec, host volume.VolumeHost, targetLocks keymutex.KeyMutex, pod *v1.Pod) (*iscsiDiskMounter, error) {
	var secret map[string]string
	readOnly, fsType, err := getISCSIVolumeInfo(spec)
	if err != nil {
		return nil, err
	}
	var podUID types.UID
	if pod != nil {
		secret, err = createSecretMap(spec, &iscsiPlugin{host: host, targetLocks: targetLocks}, pod.Namespace)
		if err != nil {
			return nil, err
		}
		podUID = pod.UID
	}
	iscsiDisk, err := createISCSIDisk(spec,
		podUID,
		&iscsiPlugin{host: host, targetLocks: targetLocks},
		&ISCSIUtil{},
		secret,
	)
	if err != nil {
		return nil, err
	}
	exec := host.GetExec(iscsiPluginName)
	// TODO: remove feature gate check after no longer needed
	if utilfeature.DefaultFeatureGate.Enabled(features.BlockVolume) {
		volumeMode, err := volumeutil.GetVolumeMode(spec)
		if err != nil {
			return nil, err
		}
		klog.V(5).Infof("iscsi: VolumeSpecToMounter volumeMode %s", volumeMode)
		return &iscsiDiskMounter{
			iscsiDisk:  iscsiDisk,
			fsType:     fsType,
			volumeMode: volumeMode,
			readOnly:   readOnly,
			mounter:    &mount.SafeFormatAndMount{Interface: host.GetMounter(iscsiPluginName), Exec: exec},
			exec:       exec,
			deviceUtil: volumeutil.NewDeviceHandler(volumeutil.NewIOHandler()),
		}, nil
	}
	return &iscsiDiskMounter{
		iscsiDisk:  iscsiDisk,
		fsType:     fsType,
		readOnly:   readOnly,
		mounter:    &mount.SafeFormatAndMount{Interface: host.GetMounter(iscsiPluginName), Exec: exec},
		exec:       exec,
		deviceUtil: volumeutil.NewDeviceHandler(volumeutil.NewIOHandler()),
	}, nil
}

func volumeSpecToUnmounter(mounter mount.Interface, host volume.VolumeHost, plugin *iscsiPlugin) *iscsiDiskUnmounter {
	exec := host.GetExec(iscsiPluginName)
	return &iscsiDiskUnmounter{
		iscsiDisk: &iscsiDisk{
			plugin: plugin,
		},
		mounter:    mounter,
		exec:       exec,
		deviceUtil: volumeutil.NewDeviceHandler(volumeutil.NewIOHandler()),
	}
}
