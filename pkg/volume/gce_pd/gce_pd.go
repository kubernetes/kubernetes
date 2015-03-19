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
	"path"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/network_volume"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{
		&gcePersistentDiskPlugin{
			NetworkVolumePlugin: &network_volume.NetworkVolumePlugin{
				PluginName: gcePersistentDiskPluginName,
				Host:       nil},
			legacyMode: false},
		&gcePersistentDiskPlugin{
			NetworkVolumePlugin: &network_volume.NetworkVolumePlugin{
				PluginName: gcePersistentDiskPluginLegacyName,
				Host:       nil},
			legacyMode: true}}
}

type gcePersistentDiskPlugin struct {
	*network_volume.NetworkVolumePlugin
	legacyMode bool // if set, plugin answers to the legacy name
}

var _ volume.VolumePlugin = &gcePersistentDiskPlugin{}

const (
	gcePersistentDiskPluginName       = "kubernetes.io/gce-pd"
	gcePersistentDiskPluginLegacyName = "gce-pd"
)

func (plugin *gcePersistentDiskPlugin) CanSupport(spec *volume.Spec) bool {
	if plugin.legacyMode {
		// Legacy mode instances can be cleaned up but not created anew.
		return false
	}

	return spec.VolumeSource.GCEPersistentDisk != nil || spec.PersistentVolumeSource.GCEPersistentDisk != nil
}

func (plugin *gcePersistentDiskPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
	}
}

func (plugin *gcePersistentDiskPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions, mounter mount.Interface) (volume.Builder, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newBuilderInternal(spec, pod.UID, &GCEDiskUtil{}, mounter)
}

func (plugin *gcePersistentDiskPlugin) newBuilderInternal(spec *volume.Spec, podUID types.UID, manager pdManager, mounter mount.Interface) (volume.Builder, error) {
	if plugin.legacyMode {
		// Legacy mode instances can be cleaned up but not created anew.
		return nil, fmt.Errorf("legacy mode: can not create new instances")
	}

	var gce *api.GCEPersistentDiskVolumeSource
	if spec.VolumeSource.GCEPersistentDisk != nil {
		gce = spec.VolumeSource.GCEPersistentDisk
	} else {
		gce = spec.PersistentVolumeSource.GCEPersistentDisk
	}

	pdName := gce.PDName
	fsType := gce.FSType
	partition := ""
	if gce.Partition != 0 {
		partition = strconv.Itoa(gce.Partition)
	}
	readOnly := gce.ReadOnly

	return &gcePersistentDisk{
		NetworkVolume: &network_volume.NetworkVolume{
			VolName: spec.Name,
			Plugin:  plugin,
			PodUID:  podUID,
			Mounter: &gceNetworkVolumeMounter{
				Interface:    mounter,
				globalPDPath: makeGlobalPDName(plugin.GetHost(), pdName),
				pdName:       pdName,
				readOnly:     readOnly,
				manager:      manager,
				diskMounter:  &gceSafeFormatAndMount{mounter, exec.New()},
				partition:    partition,
				fsType:       fsType,
			},
		},
		legacyMode: false,
	}, nil
}

func (plugin *gcePersistentDiskPlugin) NewCleaner(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newCleanerInternal(volName, podUID, &GCEDiskUtil{}, mounter)
}

func (plugin *gcePersistentDiskPlugin) newCleanerInternal(volName string, podUID types.UID, manager pdManager, mounter mount.Interface) (volume.Cleaner, error) {
	legacy := false
	if plugin.legacyMode {
		legacy = true
	}
	return &gcePersistentDisk{
		NetworkVolume: &network_volume.NetworkVolume{
			VolName: volName,
			Plugin:  plugin,
			PodUID:  podUID,
			Mounter: &gceNetworkVolumeMounter{
				Interface:   mounter,
				manager:     manager,
				diskMounter: &gceSafeFormatAndMount{mounter, exec.New()},
			},
		},
		legacyMode: legacy,
	}, nil
}

// Abstract interface to PD operations.
type pdManager interface {
	// Attaches the disk to the kubelet's host machine.
	AttachAndMountDisk(mounter *gceNetworkVolumeMounter) error
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(mounter *gceNetworkVolumeMounter) error
}

// gcePersistentDisk volumes are disk resources provided by Google Compute Engine
// that are attached to the kubelet's host machine and exposed to the pod.
type gcePersistentDisk struct {
	*network_volume.NetworkVolume

	legacyMode bool
}

// SetUp attaches the disk and bind mounts to the volume path.
func (pd *gcePersistentDisk) SetUp() error {
	if pd.legacyMode {
		return fmt.Errorf("legacy mode: can not create new instances")
	}
	return pd.NetworkVolume.SetUp()
}

// Create a global unique path for this disk to be mounted on.
func makeGlobalPDName(host volume.VolumeHost, devName string) string {
	return path.Join(host.GetPluginDir(gcePersistentDiskPluginName), "mounts", devName)
}

type gceNetworkVolumeMounter struct {
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mount.Interface
	// Globally unique path on which the disk will be mounted.
	globalPDPath string
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager pdManager
	// Unique identifier of the PD, used to find the disk resource in the provider.
	pdName string
	// Specifies whether the disk will be attached as read-only.
	readOnly bool
	// diskMounter provides the interface that is used to mount the actual block device.
	diskMounter mount.Interface
	// Specifies the partition to mount
	partition string
	// Filesystem type, optional.
	fsType string
}

// Attaches the disk and bind mounts to the volume path.
func (mounter *gceNetworkVolumeMounter) MountVolume(dest string) error {
	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	options := []string{"bind"}
	if mounter.readOnly {
		options = append(options, "ro")
	}
	return mounter.Mount(mounter.globalPDPath, dest, "", options)
}

// UnmountVolume unmounts the volume
func (mounter *gceNetworkVolumeMounter) UnmountVolume(dir string) error {
	return mounter.Unmount(dir)
}

func (mounter *gceNetworkVolumeMounter) AttachVolume() error {
	// Attach the PD to this kubelet at the global path
	return mounter.manager.AttachAndMountDisk(mounter)
}

func (mounter *gceNetworkVolumeMounter) DetachVolume(globalPath string) error {
	// pdName is not initially set for volume-cleaners, so set it here.
	mounter.pdName = path.Base(globalPath)
	mounter.globalPDPath = globalPath
	return mounter.manager.DetachDisk(mounter)
}

func (mounter *gceNetworkVolumeMounter) GetGlobalPath() string {
	return mounter.globalPDPath
}
