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

package cinder

import (
	"errors"
	"fmt"
	"os"
	"path"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/openstack"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&cinderPlugin{}}
}

type cinderPlugin struct {
	host volume.VolumeHost
	// Guarding SetUp and TearDown operations
	volumeLocks keymutex.KeyMutex
}

var _ volume.VolumePlugin = &cinderPlugin{}
var _ volume.PersistentVolumePlugin = &cinderPlugin{}
var _ volume.DeletableVolumePlugin = &cinderPlugin{}
var _ volume.ProvisionableVolumePlugin = &cinderPlugin{}

const (
	cinderVolumePluginName = "kubernetes.io/cinder"
)

func (plugin *cinderPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	plugin.volumeLocks = keymutex.NewKeyMutex()
	return nil
}

func (plugin *cinderPlugin) Name() string {
	return cinderVolumePluginName
}

func (plugin *cinderPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.Volume != nil && spec.Volume.Cinder != nil) || (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Cinder != nil)
}

func (plugin *cinderPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

func (plugin *cinderPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Builder, error) {
	return plugin.newBuilderInternal(spec, pod.UID, &CinderDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *cinderPlugin) newBuilderInternal(spec *volume.Spec, podUID types.UID, manager cdManager, mounter mount.Interface) (volume.Builder, error) {
	var cinder *api.CinderVolumeSource
	if spec.Volume != nil && spec.Volume.Cinder != nil {
		cinder = spec.Volume.Cinder
	} else {
		cinder = spec.PersistentVolume.Spec.Cinder
	}

	pdName := cinder.VolumeID
	fsType := cinder.FSType
	readOnly := cinder.ReadOnly

	return &cinderVolumeBuilder{
		cinderVolume: &cinderVolume{
			podUID:  podUID,
			volName: spec.Name(),
			pdName:  pdName,
			mounter: mounter,
			manager: manager,
			plugin:  plugin,
		},
		fsType:             fsType,
		readOnly:           readOnly,
		blockDeviceMounter: &mount.SafeFormatAndMount{mounter, exec.New()}}, nil
}

func (plugin *cinderPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	return plugin.newCleanerInternal(volName, podUID, &CinderDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *cinderPlugin) newCleanerInternal(volName string, podUID types.UID, manager cdManager, mounter mount.Interface) (volume.Cleaner, error) {
	return &cinderVolumeCleaner{
		&cinderVolume{
			podUID:  podUID,
			volName: volName,
			manager: manager,
			mounter: mounter,
			plugin:  plugin,
		}}, nil
}

func (plugin *cinderPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterInternal(spec, &CinderDiskUtil{})
}

func (plugin *cinderPlugin) newDeleterInternal(spec *volume.Spec, manager cdManager) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Cinder == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.Cinder is nil")
	}
	return &cinderVolumeDeleter{
		&cinderVolume{
			volName: spec.Name(),
			pdName:  spec.PersistentVolume.Spec.Cinder.VolumeID,
			manager: manager,
			plugin:  plugin,
		}}, nil
}

func (plugin *cinderPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	if len(options.AccessModes) == 0 {
		options.AccessModes = plugin.GetAccessModes()
	}
	return plugin.newProvisionerInternal(options, &CinderDiskUtil{})
}

func (plugin *cinderPlugin) newProvisionerInternal(options volume.VolumeOptions, manager cdManager) (volume.Provisioner, error) {
	return &cinderVolumeProvisioner{
		cinderVolume: &cinderVolume{
			manager: manager,
			plugin:  plugin,
		},
		options: options,
	}, nil
}

func (plugin *cinderPlugin) getCloudProvider() (*openstack.OpenStack, error) {
	cloud := plugin.host.GetCloudProvider()
	if cloud == nil {
		glog.Errorf("Cloud provider not initialized properly")
		return nil, errors.New("Cloud provider not initialized properly")
	}

	os := cloud.(*openstack.OpenStack)
	if os == nil {
		return nil, errors.New("Invalid cloud provider: expected OpenStack")
	}
	return os, nil
}

// Abstract interface to PD operations.
type cdManager interface {
	// Attaches the disk to the kubelet's host machine.
	AttachDisk(builder *cinderVolumeBuilder, globalPDPath string) error
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(cleaner *cinderVolumeCleaner) error
	// Creates a volume
	CreateVolume(provisioner *cinderVolumeProvisioner) (volumeID string, volumeSizeGB int, err error)
	// Deletes a volume
	DeleteVolume(deleter *cinderVolumeDeleter) error
}

var _ volume.Builder = &cinderVolumeBuilder{}

type cinderVolumeBuilder struct {
	*cinderVolume
	fsType             string
	readOnly           bool
	blockDeviceMounter *mount.SafeFormatAndMount
}

// cinderPersistentDisk volumes are disk resources provided by C3
// that are attached to the kubelet's host machine and exposed to the pod.
type cinderVolume struct {
	volName string
	podUID  types.UID
	// Unique identifier of the volume, used to find the disk resource in the provider.
	pdName string
	// Filesystem type, optional.
	fsType string
	// Specifies the partition to mount
	//partition string
	// Specifies whether the disk will be attached as read-only.
	readOnly bool
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager cdManager
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	// diskMounter provides the interface that is used to mount the actual block device.
	blockDeviceMounter mount.Interface
	plugin             *cinderPlugin
	volume.MetricsNil
}

func detachDiskLogError(cd *cinderVolume) {
	err := cd.manager.DetachDisk(&cinderVolumeCleaner{cd})
	if err != nil {
		glog.Warningf("Failed to detach disk: %v (%v)", cd, err)
	}
}

func (b *cinderVolumeBuilder) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: true,
	}
}

func (b *cinderVolumeBuilder) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *cinderVolumeBuilder) SetUpAt(dir string, fsGroup *int64) error {
	glog.V(5).Infof("Cinder SetUp %s to %s", b.pdName, dir)

	b.plugin.volumeLocks.LockKey(b.pdName)
	defer b.plugin.volumeLocks.UnlockKey(b.pdName)

	// TODO: handle failed mounts here.
	notmnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	if err != nil && !os.IsNotExist(err) {
		glog.V(4).Infof("IsLikelyNotMountPoint failed: %v", err)
		return err
	}
	if !notmnt {
		glog.V(4).Infof("Something is already mounted to target %s", dir)
		return nil
	}
	globalPDPath := makeGlobalPDName(b.plugin.host, b.pdName)
	if err := b.manager.AttachDisk(b, globalPDPath); err != nil {
		glog.V(4).Infof("AttachDisk failed: %v", err)
		return err
	}
	glog.V(3).Infof("Cinder volume %s attached", b.pdName)

	options := []string{"bind"}
	if b.readOnly {
		options = append(options, "ro")
	}

	if err := os.MkdirAll(dir, 0750); err != nil {
		// TODO: we should really eject the attach/detach out into its own control loop.
		glog.V(4).Infof("Could not create directory %s: %v", dir, err)
		detachDiskLogError(b.cinderVolume)
		return err
	}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	err = b.mounter.Mount(globalPDPath, dir, "", options)
	if err != nil {
		glog.V(4).Infof("Mount failed: %v", err)
		notmnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil {
			glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
			return err
		}
		if !notmnt {
			if mntErr = b.mounter.Unmount(dir); mntErr != nil {
				glog.Errorf("Failed to unmount: %v", mntErr)
				return err
			}
			notmnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
				return err
			}
			if !notmnt {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				glog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", b.GetPath())
				return err
			}
		}
		os.Remove(dir)
		// TODO: we should really eject the attach/detach out into its own control loop.
		detachDiskLogError(b.cinderVolume)
		return err
	}

	if !b.readOnly {
		volume.SetVolumeOwnership(b, fsGroup)
	}
	glog.V(3).Infof("Cinder volume %s mounted to %s", b.pdName, dir)

	return nil
}

func makeGlobalPDName(host volume.VolumeHost, devName string) string {
	return path.Join(host.GetPluginDir(cinderVolumePluginName), "mounts", devName)
}

func (cd *cinderVolume) GetPath() string {
	name := cinderVolumePluginName
	return cd.plugin.host.GetPodVolumeDir(cd.podUID, strings.EscapeQualifiedNameForDisk(name), cd.volName)
}

type cinderVolumeCleaner struct {
	*cinderVolume
}

var _ volume.Cleaner = &cinderVolumeCleaner{}

func (c *cinderVolumeCleaner) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *cinderVolumeCleaner) TearDownAt(dir string) error {
	glog.V(5).Infof("Cinder TearDown of %s", dir)
	notmnt, err := c.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		glog.V(4).Infof("IsLikelyNotMountPoint check failed: %v", err)
		return err
	}
	if notmnt {
		glog.V(4).Infof("Nothing is mounted to %s, ignoring", dir)
		return os.Remove(dir)
	}

	// Find Cinder volumeID to lock the right volume
	// TODO: refactor VolumePlugin.NewCleaner to get full volume.Spec just like
	// NewBuilder. We could then find volumeID there without probing MountRefs.
	refs, err := mount.GetMountRefs(c.mounter, dir)
	if err != nil {
		glog.V(4).Infof("GetMountRefs failed: %v", err)
		return err
	}
	if len(refs) == 0 {
		glog.V(4).Infof("Directory %s is not mounted", dir)
		return fmt.Errorf("directory %s is not mounted", dir)
	}
	c.pdName = path.Base(refs[0])
	glog.V(4).Infof("Found volume %s mounted to %s", c.pdName, dir)

	// lock the volume (and thus wait for any concurrrent SetUpAt to finish)
	c.plugin.volumeLocks.LockKey(c.pdName)
	defer c.plugin.volumeLocks.UnlockKey(c.pdName)

	// Reload list of references, there might be SetUpAt finished in the meantime
	refs, err = mount.GetMountRefs(c.mounter, dir)
	if err != nil {
		glog.V(4).Infof("GetMountRefs failed: %v", err)
		return err
	}
	if err := c.mounter.Unmount(dir); err != nil {
		glog.V(4).Infof("Unmount failed: %v", err)
		return err
	}
	glog.V(3).Infof("Successfully unmounted: %s\n", dir)

	// If refCount is 1, then all bind mounts have been removed, and the
	// remaining reference is the global mount. It is safe to detach.
	if len(refs) == 1 {
		if err := c.manager.DetachDisk(c); err != nil {
			glog.V(4).Infof("DetachDisk failed: %v", err)
			return err
		}
		glog.V(3).Infof("Volume %s detached", c.pdName)
	}
	notmnt, mntErr := c.mounter.IsLikelyNotMountPoint(dir)
	if mntErr != nil {
		glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
		return err
	}
	if notmnt {
		if err := os.Remove(dir); err != nil {
			glog.V(4).Infof("Failed to remove directory after unmount: %v", err)
			return err
		}
	}
	return nil
}

type cinderVolumeDeleter struct {
	*cinderVolume
}

var _ volume.Deleter = &cinderVolumeDeleter{}

func (r *cinderVolumeDeleter) GetPath() string {
	name := cinderVolumePluginName
	return r.plugin.host.GetPodVolumeDir(r.podUID, strings.EscapeQualifiedNameForDisk(name), r.volName)
}

func (r *cinderVolumeDeleter) Delete() error {
	return r.manager.DeleteVolume(r)
}

type cinderVolumeProvisioner struct {
	*cinderVolume
	options volume.VolumeOptions
}

var _ volume.Provisioner = &cinderVolumeProvisioner{}

func (c *cinderVolumeProvisioner) Provision(pv *api.PersistentVolume) error {
	volumeID, sizeGB, err := c.manager.CreateVolume(c)
	if err != nil {
		return err
	}
	pv.Spec.PersistentVolumeSource.Cinder.VolumeID = volumeID
	pv.Spec.Capacity = api.ResourceList{
		api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
	}
	return nil
}

func (c *cinderVolumeProvisioner) NewPersistentVolumeTemplate() (*api.PersistentVolume, error) {
	// Provide dummy api.PersistentVolume.Spec, it will be filled in
	// cinderVolumeProvisioner.Provision()
	return &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pv-cinder-",
			Labels:       map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "cinder-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: c.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   c.options.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): c.options.Capacity,
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				Cinder: &api.CinderVolumeSource{
					VolumeID: "dummy",
					FSType:   "ext4",
					ReadOnly: false,
				},
			},
		},
	}, nil

}
