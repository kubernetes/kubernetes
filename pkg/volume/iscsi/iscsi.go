/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/network_volume"
	"github.com/golang/glog"
)

// Abstract interface to disk operations.
type diskManager interface {
	// Attaches the disk to the kubelet's host machine.
	AttachDisk(*iscsiNetworkVolumeMounter) error
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(*iscsiNetworkVolumeMounter) error
}

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&iscsiPlugin{
		NetworkVolumePlugin: &network_volume.NetworkVolumePlugin{
			PluginName: iscsiPluginName,
			Host:       nil},
		exe: exec.New()}}
}

type iscsiPlugin struct {
	*network_volume.NetworkVolumePlugin
	exe exec.Interface
}

var _ volume.VolumePlugin = &iscsiPlugin{}

const (
	iscsiPluginName = "kubernetes.io/iscsi"
)

func (plugin *iscsiPlugin) CanSupport(spec *volume.Spec) bool {
	if spec.VolumeSource.ISCSI == nil && spec.PersistentVolumeSource.ISCSI == nil {
		return false
	}
	// TODO:  turn this into a func so CanSupport can be unit tested without
	// having to make system calls
	// see if iscsiadm is there
	_, err := plugin.execCommand("iscsiadm", []string{"-h"})
	if err == nil {
		return true
	}

	return false
}

func (plugin *iscsiPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
	}
}

func (plugin *iscsiPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions, mounter mount.Interface) (volume.Builder, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newBuilderInternal(spec, pod.UID, &ISCSIUtil{}, mounter)
}

func (plugin *iscsiPlugin) newBuilderInternal(spec *volume.Spec, podUID types.UID, manager diskManager, mounter mount.Interface) (volume.Builder, error) {
	var iscsi *api.ISCSIVolumeSource
	if spec.VolumeSource.ISCSI != nil {
		iscsi = spec.VolumeSource.ISCSI
	} else {
		iscsi = spec.PersistentVolumeSource.ISCSI
	}

	lun := strconv.Itoa(iscsi.Lun)

	return &iscsiDisk{
		NetworkVolume: &network_volume.NetworkVolume{
			VolName: spec.Name,
			PodUID:  podUID,
			Plugin:  plugin,
			Mounter: &iscsiNetworkVolumeMounter{
				Interface:    mounter,
				globalPDPath: makePDNameInternal(plugin.GetHost(), iscsi.TargetPortal, iscsi.IQN, lun),
				portal:       iscsi.TargetPortal,
				iqn:          iscsi.IQN,
				lun:          lun,
				fsType:       iscsi.FSType,
				readOnly:     iscsi.ReadOnly,
				manager:      manager,
				plugin:       plugin,
			},
		},
	}, nil
}

func (plugin *iscsiPlugin) NewCleaner(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newCleanerInternal(volName, podUID, &ISCSIUtil{}, mounter)
}

func (plugin *iscsiPlugin) newCleanerInternal(volName string, podUID types.UID, manager diskManager, mounter mount.Interface) (volume.Cleaner, error) {
	return &iscsiDisk{
		NetworkVolume: &network_volume.NetworkVolume{
			PodUID:  podUID,
			VolName: volName,
			Plugin:  plugin,
			Mounter: &iscsiNetworkVolumeMounter{
				Interface: mounter,
				manager:   manager,
				plugin:    plugin,
			},
		},
	}, nil
}

func (plugin *iscsiPlugin) execCommand(command string, args []string) ([]byte, error) {
	cmd := plugin.exe.Command(command, args...)
	return cmd.CombinedOutput()
}

type iscsiDisk struct {
	*network_volume.NetworkVolume
}

type iscsiNetworkVolumeMounter struct {
	mount.Interface
	globalPDPath string
	portal       string
	iqn          string
	readOnly     bool
	lun          string
	fsType       string
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager diskManager
	plugin  *iscsiPlugin
}

func (mounter *iscsiNetworkVolumeMounter) MountVolume(dest string) error {
	// Perform a bind mount to the full path to allow duplicate mounts of the same disk.
	options := []string{"bind"}
	if mounter.readOnly {
		options = append(options, "ro")
	}

	err := mounter.Mount(mounter.globalPDPath, dest, "", options)
	if err != nil {
		glog.Errorf("failed to bind mount:%s", mounter.globalPDPath)
		return err
	}
	// Remount is needed because the first mount does not honor
	// the ro option when passed
	// TODO: Check that remount is still needed and debug if so.
	if mounter.readOnly {
		options = []string{"remount", "ro"}
	} else {
		options = []string{"remount", "rw"}
	}
	return mounter.Mount(mounter.globalPDPath, dest, "", options)
}

func (mounter *iscsiNetworkVolumeMounter) UnmountVolume(dest string) error {
	return mounter.Unmount(dest)
}

func (mounter *iscsiNetworkVolumeMounter) AttachVolume() error {
	return mounter.manager.AttachDisk(mounter)
}

func (mounter *iscsiNetworkVolumeMounter) DetachVolume(globalPath string) error {
	mounter.globalPDPath = globalPath
	return mounter.manager.DetachDisk(mounter)
}

func (mounter *iscsiNetworkVolumeMounter) GetGlobalPath() string {
	return mounter.globalPDPath
}
