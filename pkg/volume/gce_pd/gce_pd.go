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
	"os"
	"path"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/golang/glog"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&gcePersistentDiskPlugin{nil, false}, &gcePersistentDiskPlugin{nil, true}}
}

type gcePersistentDiskPlugin struct {
	host       volume.VolumeHost
	legacyMode bool // if set, plugin answers to the legacy name
}

var _ volume.VolumePlugin = &gcePersistentDiskPlugin{}

const (
	gcePersistentDiskPluginName       = "kubernetes.io/gce-pd"
	gcePersistentDiskPluginLegacyName = "gce-pd"
)

func (plugin *gcePersistentDiskPlugin) Init(host volume.VolumeHost) {
	plugin.host = host
}

func (plugin *gcePersistentDiskPlugin) Name() string {
	if plugin.legacyMode {
		return gcePersistentDiskPluginLegacyName
	}
	return gcePersistentDiskPluginName
}

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
		podUID:      podUID,
		volName:     spec.Name,
		pdName:      pdName,
		fsType:      fsType,
		partition:   partition,
		readOnly:    readOnly,
		manager:     manager,
		mounter:     mounter,
		diskMounter: &gceSafeFormatAndMount{mounter, exec.New()},
		plugin:      plugin,
		legacyMode:  false,
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
		podUID:      podUID,
		volName:     volName,
		manager:     manager,
		mounter:     mounter,
		diskMounter: &gceSafeFormatAndMount{mounter, exec.New()},
		plugin:      plugin,
		legacyMode:  legacy,
	}, nil
}

// Abstract interface to PD operations.
type pdManager interface {
	// Attaches the disk to the kubelet's host machine.
	AttachAndMountDisk(pd *gcePersistentDisk, globalPDPath string) error
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(pd *gcePersistentDisk) error
}

// gcePersistentDisk volumes are disk resources provided by Google Compute Engine
// that are attached to the kubelet's host machine and exposed to the pod.
type gcePersistentDisk struct {
	volName string
	podUID  types.UID
	// Unique identifier of the PD, used to find the disk resource in the provider.
	pdName string
	// Filesystem type, optional.
	fsType string
	// Specifies the partition to mount
	partition string
	// Specifies whether the disk will be attached as read-only.
	readOnly bool
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager pdManager
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	// diskMounter provides the interface that is used to mount the actual block device.
	diskMounter mount.Interface
	plugin      *gcePersistentDiskPlugin
	legacyMode  bool
}

func detachDiskLogError(pd *gcePersistentDisk) {
	err := pd.manager.DetachDisk(pd)
	if err != nil {
		glog.Warningf("Failed to detach disk: %v (%v)", pd, err)
	}
}

// SetUp attaches the disk and bind mounts to the volume path.
func (pd *gcePersistentDisk) SetUp() error {
	return pd.SetUpAt(pd.GetPath())
}

// SetUpAt attaches the disk and bind mounts to the volume path.
func (pd *gcePersistentDisk) SetUpAt(dir string) error {
	if pd.legacyMode {
		return fmt.Errorf("legacy mode: can not create new instances")
	}

	// TODO: handle failed mounts here.
	mountpoint, err := pd.mounter.IsMountPoint(dir)
	glog.V(4).Infof("PersistentDisk set up: %s %v %v", dir, mountpoint, err)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if mountpoint {
		return nil
	}

	globalPDPath := makeGlobalPDName(pd.plugin.host, pd.pdName)
	if err := pd.manager.AttachAndMountDisk(pd, globalPDPath); err != nil {
		return err
	}

	if err := os.MkdirAll(dir, 0750); err != nil {
		// TODO: we should really eject the attach/detach out into its own control loop.
		detachDiskLogError(pd)
		return err
	}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	options := []string{"bind"}
	if pd.readOnly {
		options = append(options, "ro")
	}
	err = pd.mounter.Mount(globalPDPath, dir, "", options)
	if err != nil {
		mountpoint, mntErr := pd.mounter.IsMountPoint(dir)
		if mntErr != nil {
			glog.Errorf("isMountpoint check failed: %v", mntErr)
			return err
		}
		if mountpoint {
			if mntErr = pd.mounter.Unmount(dir); mntErr != nil {
				glog.Errorf("Failed to unmount: %v", mntErr)
				return err
			}
			mountpoint, mntErr := pd.mounter.IsMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("isMountpoint check failed: %v", mntErr)
				return err
			}
			if mountpoint {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				glog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", dir)
				return err
			}
		}
		os.Remove(dir)
		// TODO: we should really eject the attach/detach out into its own control loop.
		detachDiskLogError(pd)
		return err
	}

	return nil
}

func makeGlobalPDName(host volume.VolumeHost, devName string) string {
	return path.Join(host.GetPluginDir(gcePersistentDiskPluginName), "mounts", devName)
}

func (pd *gcePersistentDisk) GetPath() string {
	name := gcePersistentDiskPluginName
	if pd.legacyMode {
		name = gcePersistentDiskPluginLegacyName
	}
	return pd.plugin.host.GetPodVolumeDir(pd.podUID, util.EscapeQualifiedNameForDisk(name), pd.volName)
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (pd *gcePersistentDisk) TearDown() error {
	return pd.TearDownAt(pd.GetPath())
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (pd *gcePersistentDisk) TearDownAt(dir string) error {
	mountpoint, err := pd.mounter.IsMountPoint(dir)
	if err != nil {
		return err
	}
	if !mountpoint {
		return os.Remove(dir)
	}

	refs, err := mount.GetMountRefs(pd.mounter, dir)
	if err != nil {
		return err
	}
	// Unmount the bind-mount inside this pod
	if err := pd.mounter.Unmount(dir); err != nil {
		return err
	}
	// If len(refs) is 1, then all bind mounts have been removed, and the
	// remaining reference is the global mount. It is safe to detach.
	if len(refs) == 1 {
		// pd.pdName is not initially set for volume-cleaners, so set it here.
		pd.pdName = path.Base(refs[0])
		if err := pd.manager.DetachDisk(pd); err != nil {
			return err
		}
	}
	mountpoint, mntErr := pd.mounter.IsMountPoint(dir)
	if mntErr != nil {
		glog.Errorf("isMountpoint check failed: %v", mntErr)
		return err
	}
	if !mountpoint {
		if err := os.Remove(dir); err != nil {
			return err
		}
	}
	return nil
}
