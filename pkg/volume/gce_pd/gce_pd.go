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
	"os"
	"path"
	"strconv"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&gcePersistentDiskPlugin{nil}}
}

type gcePersistentDiskPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &gcePersistentDiskPlugin{}
var _ volume.PersistentVolumePlugin = &gcePersistentDiskPlugin{}

const (
	gcePersistentDiskPluginName = "kubernetes.io/gce-pd"
)

func (plugin *gcePersistentDiskPlugin) Init(host volume.VolumeHost) {
	plugin.host = host
}

func (plugin *gcePersistentDiskPlugin) Name() string {
	return gcePersistentDiskPluginName
}

func (plugin *gcePersistentDiskPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.GCEPersistentDisk != nil) ||
		(spec.Volume != nil && spec.Volume.GCEPersistentDisk != nil)
}

func (plugin *gcePersistentDiskPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
	}
}

func (plugin *gcePersistentDiskPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Builder, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newBuilderInternal(spec, pod.UID, &GCEDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *gcePersistentDiskPlugin) newBuilderInternal(spec *volume.Spec, podUID types.UID, manager pdManager, mounter mount.Interface) (volume.Builder, error) {
	// GCEPDs used directly in a pod have a ReadOnly flag set by the pod author.
	// GCEPDs used as a PersistentVolume gets the ReadOnly flag indirectly through the persistent-claim volume used to mount the PV
	var readOnly bool

	var gce *api.GCEPersistentDiskVolumeSource
	if spec.Volume != nil && spec.Volume.GCEPersistentDisk != nil {
		gce = spec.Volume.GCEPersistentDisk
		readOnly = gce.ReadOnly
	} else {
		gce = spec.PersistentVolume.Spec.GCEPersistentDisk
		readOnly = spec.ReadOnly
	}

	pdName := gce.PDName
	fsType := gce.FSType
	partition := ""
	if gce.Partition != 0 {
		partition = strconv.Itoa(gce.Partition)
	}

	return &gcePersistentDiskBuilder{
		gcePersistentDisk: &gcePersistentDisk{
			podUID:    podUID,
			volName:   spec.Name(),
			pdName:    pdName,
			partition: partition,
			mounter:   mounter,
			manager:   manager,
			plugin:    plugin,
		},
		fsType:      fsType,
		readOnly:    readOnly,
		diskMounter: &mount.SafeFormatAndMount{mounter, exec.New()}}, nil
}

func (plugin *gcePersistentDiskPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newCleanerInternal(volName, podUID, &GCEDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *gcePersistentDiskPlugin) newCleanerInternal(volName string, podUID types.UID, manager pdManager, mounter mount.Interface) (volume.Cleaner, error) {
	return &gcePersistentDiskCleaner{&gcePersistentDisk{
		podUID:  podUID,
		volName: volName,
		manager: manager,
		mounter: mounter,
		plugin:  plugin,
	}}, nil
}

// Abstract interface to PD operations.
type pdManager interface {
	// Attaches the disk to the kubelet's host machine.
	AttachAndMountDisk(b *gcePersistentDiskBuilder, globalPDPath string) error
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(c *gcePersistentDiskCleaner) error
}

// gcePersistentDisk volumes are disk resources provided by Google Compute Engine
// that are attached to the kubelet's host machine and exposed to the pod.
type gcePersistentDisk struct {
	volName string
	podUID  types.UID
	// Unique identifier of the PD, used to find the disk resource in the provider.
	pdName string
	// Specifies the partition to mount
	partition string
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager pdManager
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	plugin  *gcePersistentDiskPlugin
}

func detachDiskLogError(pd *gcePersistentDisk) {
	err := pd.manager.DetachDisk(&gcePersistentDiskCleaner{pd})
	if err != nil {
		glog.Warningf("Failed to detach disk: %v (%v)", pd, err)
	}
}

type gcePersistentDiskBuilder struct {
	*gcePersistentDisk
	// Filesystem type, optional.
	fsType string
	// Specifies whether the disk will be attached as read-only.
	readOnly bool
	// diskMounter provides the interface that is used to mount the actual block device.
	diskMounter mount.Interface
}

var _ volume.Builder = &gcePersistentDiskBuilder{}

func (_ *gcePersistentDiskBuilder) SupportsOwnershipManagement() bool {
	return true
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *gcePersistentDiskBuilder) SetUp() error {
	return b.SetUpAt(b.GetPath())
}

// SetUpAt attaches the disk and bind mounts to the volume path.
func (b *gcePersistentDiskBuilder) SetUpAt(dir string) error {
	// TODO: handle failed mounts here.
	notMnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	glog.V(4).Infof("PersistentDisk set up: %s %v %v", dir, !notMnt, err)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if !notMnt {
		return nil
	}

	globalPDPath := makeGlobalPDName(b.plugin.host, b.pdName)
	if err := b.manager.AttachAndMountDisk(b, globalPDPath); err != nil {
		return err
	}

	if err := os.MkdirAll(dir, 0750); err != nil {
		// TODO: we should really eject the attach/detach out into its own control loop.
		detachDiskLogError(b.gcePersistentDisk)
		return err
	}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	options := []string{"bind"}
	if b.readOnly {
		options = append(options, "ro")
	}
	err = b.mounter.Mount(globalPDPath, dir, "", options)
	if err != nil {
		notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil {
			glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
			return err
		}
		if !notMnt {
			if mntErr = b.mounter.Unmount(dir); mntErr != nil {
				glog.Errorf("Failed to unmount: %v", mntErr)
				return err
			}
			notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
				return err
			}
			if !notMnt {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				glog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", dir)
				return err
			}
		}
		os.Remove(dir)
		// TODO: we should really eject the attach/detach out into its own control loop.
		detachDiskLogError(b.gcePersistentDisk)
		return err
	}

	return nil
}

func (b *gcePersistentDiskBuilder) IsReadOnly() bool {
	return b.readOnly
}

func makeGlobalPDName(host volume.VolumeHost, devName string) string {
	return path.Join(host.GetPluginDir(gcePersistentDiskPluginName), "mounts", devName)
}

func (pd *gcePersistentDisk) GetPath() string {
	name := gcePersistentDiskPluginName
	return pd.plugin.host.GetPodVolumeDir(pd.podUID, util.EscapeQualifiedNameForDisk(name), pd.volName)
}

type gcePersistentDiskCleaner struct {
	*gcePersistentDisk
}

var _ volume.Cleaner = &gcePersistentDiskCleaner{}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *gcePersistentDiskCleaner) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *gcePersistentDiskCleaner) TearDownAt(dir string) error {
	notMnt, err := c.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		return err
	}
	if notMnt {
		return os.Remove(dir)
	}

	refs, err := mount.GetMountRefs(c.mounter, dir)
	if err != nil {
		return err
	}
	// Unmount the bind-mount inside this pod
	if err := c.mounter.Unmount(dir); err != nil {
		return err
	}
	// If len(refs) is 1, then all bind mounts have been removed, and the
	// remaining reference is the global mount. It is safe to detach.
	if len(refs) == 1 {
		// c.pdName is not initially set for volume-cleaners, so set it here.
		c.pdName = path.Base(refs[0])
		if err := c.manager.DetachDisk(c); err != nil {
			return err
		}
	}
	notMnt, mntErr := c.mounter.IsLikelyNotMountPoint(dir)
	if mntErr != nil {
		glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
		return err
	}
	if notMnt {
		if err := os.Remove(dir); err != nil {
			return err
		}
	}
	return nil
}
