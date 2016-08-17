/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/openstack"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/rackspace"
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

type CinderProvider interface {
	AttachDisk(instanceID string, diskName string) (string, error)
	DetachDisk(instanceID string, partialDiskId string) error
	DeleteVolume(volumeName string) error
	CreateVolume(name string, size int, tags *map[string]string) (volumeName string, err error)
	GetDevicePath(diskId string) string
	InstanceID() (string, error)
	GetAttachmentDiskPath(instanceID string, diskName string) (string, error)
	DiskIsAttached(diskName, instanceID string) (bool, error)
	Instances() (cloudprovider.Instances, bool)
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

func (plugin *cinderPlugin) GetPluginName() string {
	return cinderVolumePluginName
}

func (plugin *cinderPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.VolumeID, nil
}

func (plugin *cinderPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.Volume != nil && spec.Volume.Cinder != nil) || (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Cinder != nil)
}

func (plugin *cinderPlugin) RequiresRemount() bool {
	return false
}

func (plugin *cinderPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

func (plugin *cinderPlugin) NewMounter(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod.UID, &CinderDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *cinderPlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager cdManager, mounter mount.Interface) (volume.Mounter, error) {
	cinder, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	pdName := cinder.VolumeID
	fsType := cinder.FSType

	return &cinderVolumeMounter{
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
		blockDeviceMounter: &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}}, nil
}

func (plugin *cinderPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, &CinderDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *cinderPlugin) newUnmounterInternal(volName string, podUID types.UID, manager cdManager, mounter mount.Interface) (volume.Unmounter, error) {
	return &cinderVolumeUnmounter{
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

func getCloudProvider(cloudProvider cloudprovider.Interface) (CinderProvider, error) {
	if cloud, ok := cloudProvider.(*rackspace.Rackspace); ok && cloud != nil {
		return cloud, nil
	}
	if cloud, ok := cloudProvider.(*openstack.OpenStack); ok && cloud != nil {
		return cloud, nil
	}
	return nil, fmt.Errorf("wrong cloud type")
}

func (plugin *cinderPlugin) getCloudProvider() (CinderProvider, error) {
	cloud := plugin.host.GetCloudProvider()
	if cloud == nil {
		glog.Errorf("Cloud provider not initialized properly")
		return nil, errors.New("Cloud provider not initialized properly")
	}

	switch cloud := cloud.(type) {
	case *rackspace.Rackspace:
		return cloud, nil
	case *openstack.OpenStack:
		return cloud, nil
	default:
		return nil, errors.New("Invalid cloud provider: expected OpenStack or Rackspace.")
	}
}

func (plugin *cinderPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	mounter := plugin.host.GetMounter()
	pluginDir := plugin.host.GetPluginDir(plugin.GetPluginName())
	sourceName, err := mounter.GetDeviceNameFromMount(mountPath, pluginDir)
	if err != nil {
		return nil, err
	}
	glog.V(4).Infof("Found volume %s mounted to %s", sourceName, mountPath)
	cinderVolume := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			Cinder: &api.CinderVolumeSource{
				VolumeID: sourceName,
			},
		},
	}
	return volume.NewSpecFromVolume(cinderVolume), nil
}

// Abstract interface to PD operations.
type cdManager interface {
	// Attaches the disk to the kubelet's host machine.
	AttachDisk(mounter *cinderVolumeMounter, globalPDPath string) error
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(unmounter *cinderVolumeUnmounter) error
	// Creates a volume
	CreateVolume(provisioner *cinderVolumeProvisioner) (volumeID string, volumeSizeGB int, err error)
	// Deletes a volume
	DeleteVolume(deleter *cinderVolumeDeleter) error
}

var _ volume.Mounter = &cinderVolumeMounter{}

type cinderVolumeMounter struct {
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
	err := cd.manager.DetachDisk(&cinderVolumeUnmounter{cd})
	if err != nil {
		glog.Warningf("Failed to detach disk: %v (%v)", cd, err)
	}
}

func (b *cinderVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: true,
	}
}

func (b *cinderVolumeMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

// SetUp bind mounts to the volume path.
func (b *cinderVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {
	glog.V(5).Infof("Cinder SetUp %s to %s", b.pdName, dir)

	b.plugin.volumeLocks.LockKey(b.pdName)
	defer b.plugin.volumeLocks.UnlockKey(b.pdName)

	// TODO: handle failed mounts here.
	notmnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("Cannot validate mount point: %s %v", dir, err)
		return err
	}
	if !notmnt {
		glog.V(4).Infof("Something is already mounted to target %s", dir)
		return nil
	}
	globalPDPath := makeGlobalPDName(b.plugin.host, b.pdName)

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
	glog.V(4).Infof("Attempting to mount cinder volume %s to %s with options %v", b.pdName, dir, options)
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
		glog.Errorf("Failed to mount %s: %v", dir, err)
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

type cinderVolumeUnmounter struct {
	*cinderVolume
}

var _ volume.Unmounter = &cinderVolumeUnmounter{}

func (c *cinderVolumeUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *cinderVolumeUnmounter) TearDownAt(dir string) error {
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
	// TODO: refactor VolumePlugin.NewUnmounter to get full volume.Spec just like
	// NewMounter. We could then find volumeID there without probing MountRefs.
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

func (c *cinderVolumeProvisioner) Provision() (*api.PersistentVolume, error) {
	volumeID, sizeGB, err := c.manager.CreateVolume(c)
	if err != nil {
		return nil, err
	}

	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:   c.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "cinder-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: c.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   c.options.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				Cinder: &api.CinderVolumeSource{
					VolumeID: volumeID,
					FSType:   "ext4",
					ReadOnly: false,
				},
			},
		},
	}
	return pv, nil
}

func getVolumeSource(spec *volume.Spec) (*api.CinderVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.Cinder != nil {
		return spec.Volume.Cinder, spec.Volume.Cinder.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.Cinder != nil {
		return spec.PersistentVolume.Spec.Cinder, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a Cinder volume type")
}
