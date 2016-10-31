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

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/digitalocean"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&doVolumePlugin{}}
}

type DoProvider interface {
	AttachVolume(instanceID int, volumeID string) (string, error)
	DetachVolume(instanceID int, partialVolumeId string) error
	DeleteVolume(volumeName string) error
	CreateVolume(region string, name string, description string, sizeGigaBytes int64) (volumeName string, err error)
	GetDevicePath(diskId string) string
	LocalInstanceID() (string, error)
	GetAttachmentVolumePath(instanceID int, volumeName string) (string, error)
	VolumeIsAttached(volumeName string, instanceID int) (bool, error)
	VolumesAreAttached(volumeNames []string, instanceID int) (map[string]bool, error)
	Instances() (cloudprovider.Instances, bool)
	GetRegion() string
}

type doVolumePlugin struct {
	host volume.VolumeHost
	// Guarding SetUp and TearDown operations
	volumeLocks keymutex.KeyMutex
}

var _ volume.VolumePlugin = &doVolumePlugin{}
var _ volume.PersistentVolumePlugin = &doVolumePlugin{}
var _ volume.DeletableVolumePlugin = &doVolumePlugin{}
var _ volume.ProvisionableVolumePlugin = &doVolumePlugin{}

const (
	doVolumePluginName = "kubernetes.io/digitalocean-volume"
)

// Abstract interface to PD operations.
type cdManager interface {
	// Attaches the disk to the kubelet's host machine.
	AttachVolume(mounter *doVolumeMounter, globalPDPath string) error
	// Detaches the disk from the kubelet's host machine.
	DetachVolume(unmounter *doVolumeUnmounter) error
	// Creates a volume
	CreateVolume(provisioner *doVolumeProvisioner) (volumeID string, volumeSizeGB int, err error)
	// Deletes a volume
	DeleteVolume(deleter *doVolumeDeleter) error
}

func (plugin *doVolumePlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	plugin.volumeLocks = keymutex.NewKeyMutex()
	return nil
}

func (plugin *doVolumePlugin) GetPluginName() string {
	return doVolumePluginName
}

func (plugin *doVolumePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.VolumeID, nil
}

func (plugin *doVolumePlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.Volume != nil && spec.Volume.DigitalOceanVolume != nil) || (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.DigitalOceanVolume != nil)
}

func (plugin *doVolumePlugin) RequiresRemount() bool {
	return false
}

func (plugin *doVolumePlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

func (plugin *doVolumePlugin) NewMounter(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod.UID, &DoDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *doVolumePlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager cdManager, mounter mount.Interface) (volume.Mounter, error) {
	d, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	pdName := d.VolumeID
	fsType := d.FSType

	return &doVolumeMounter{
		doVolume: &doVolume{
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

func (plugin *doVolumePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, &DoDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *doVolumePlugin) newUnmounterInternal(volName string, podUID types.UID, manager cdManager, mounter mount.Interface) (volume.Unmounter, error) {
	return &doVolumeUnmounter{
		&doVolume{
			podUID:  podUID,
			volName: volName,
			manager: manager,
			mounter: mounter,
			plugin:  plugin,
		}}, nil
}

func (plugin *doVolumePlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	mounter := plugin.host.GetMounter()
	pluginDir := plugin.host.GetPluginDir(plugin.GetPluginName())
	sourceName, err := mounter.GetDeviceNameFromMount(mountPath, pluginDir)
	if err != nil {
		return nil, err
	}
	glog.V(4).Infof("Found volume %s mounted to %s", sourceName, mountPath)
	doVolume := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			DigitalOceanVolume: &api.DigitalOceanVolumeSource{
				VolumeID: sourceName,
			},
		},
	}
	return volume.NewSpecFromVolume(doVolume), nil
}

var _ volume.Mounter = &doVolumeMounter{}

type doVolumeMounter struct {
	*doVolume
	fsType             string
	readOnly           bool
	blockDeviceMounter *mount.SafeFormatAndMount
}

// DigitalOcean volumes represent a bare host file or directory mount of an DigitalOcean export.
type doVolume struct {
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
	plugin             *doVolumePlugin
	volume.MetricsNil
}

func (doVolume *doVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        doVolume.readOnly,
		Managed:         !doVolume.readOnly,
		SupportsSELinux: true,
	}
}

func (b *doVolumeMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *doVolumeMounter) CanMount() error {
	return nil
}

// SetUp bind mounts to the volume path.
func (b *doVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {
	glog.V(5).Infof("DigitalOcean Volume SetUp %s to %s", b.pdName, dir)

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
		detachDiskLogError(b.doVolume)
		return err
	}
	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	glog.V(4).Infof("Attempting to mount digitalocean volume %s to %s with options %v", b.pdName, dir, options)
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
		detachDiskLogError(b.doVolume)
		glog.Errorf("Failed to mount %s: %v", dir, err)
		return err
	}

	if !b.readOnly {
		volume.SetVolumeOwnership(b, fsGroup)
	}
	glog.V(3).Infof("DigitalOcean volume %s mounted to %s", b.pdName, dir)

	return nil
}

func detachDiskLogError(d *doVolume) {
	err := d.manager.DetachVolume(&doVolumeUnmounter{d})
	if err != nil {
		glog.Warningf("Failed to detach disk: %v (%v)", d, err)
	}
}

type doVolumeUnmounter struct {
	*doVolume
}

var _ volume.Unmounter = &doVolumeUnmounter{}

// TearDown unmounts the bind mount
func (doVolumeVolume *doVolumeUnmounter) TearDown() error {
	return doVolumeVolume.TearDownAt(doVolumeVolume.GetPath())
}

// GatePath creates global mount path
func (doVolume *doVolume) GetPath() string {
	name := doVolumePluginName
	return doVolume.plugin.host.GetPodVolumeDir(doVolume.podUID, strings.EscapeQualifiedNameForDisk(name), doVolume.volName)
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *doVolumeUnmounter) TearDownAt(dir string) error {
	glog.V(5).Infof("DigitalOcean TearDown of %s", dir)
	notmnt, err := c.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		glog.V(4).Infof("IsLikelyNotMountPoint check failed: %v", err)
		return err
	}
	if notmnt {
		glog.V(4).Infof("Nothing is mounted to %s, ignoring", dir)
		return os.Remove(dir)
	}

	// Find DigitalOcean volumeID to lock the right volume
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

func getVolumeSource(spec *volume.Spec) (*api.DigitalOceanVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.DigitalOceanVolume != nil {
		return spec.Volume.DigitalOceanVolume, spec.Volume.DigitalOceanVolume.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.DigitalOceanVolume != nil {
		return spec.PersistentVolume.Spec.DigitalOceanVolume, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a DigitalOcean volume type")
}

func getCloudProvider(cloudProvider cloudprovider.Interface) (DoProvider, error) {
	if cloud, ok := cloudProvider.(*digitalocean.DigitalOcean); ok && cloud != nil {
		return cloud, nil
	}
	return nil, fmt.Errorf("Could not initialize cloud provider")
}

// Deleter
type doVolumeDeleter struct {
	*doVolume
}

var _ volume.Deleter = &doVolumeDeleter{}

func (r *doVolumeDeleter) GetPath() string {
	name := doVolumePluginName
	return r.plugin.host.GetPodVolumeDir(r.podUID, strings.EscapeQualifiedNameForDisk(name), r.volName)
}

func (r *doVolumeDeleter) Delete() error {
	return r.manager.DeleteVolume(r)
}

// Provisioner
type doVolumeProvisioner struct {
	*doVolume
	options volume.VolumeOptions
	plugin  *doVolumePlugin
}

var _ volume.Provisioner = &doVolumeProvisioner{}

func (d *doVolumeProvisioner) Provision() (*api.PersistentVolume, error) {
	volumeID, sizeGB, err := d.manager.CreateVolume(d)
	if err != nil {
		return nil, err
	}

	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:   d.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "digitalocean-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: d.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   d.options.PVC.Spec.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				DigitalOceanVolume: &api.DigitalOceanVolumeSource{
					VolumeID: volumeID,
					FSType:   "ext4",
					ReadOnly: false,
				},
			},
		},
	}
	if len(d.options.PVC.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = d.plugin.GetAccessModes()
	}

	return pv, nil
}

func (plugin *doVolumePlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterInternal(spec, &DoDiskUtil{})
}
func (plugin *doVolumePlugin) newDeleterInternal(spec *volume.Spec, manager cdManager) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.DigitalOceanVolume == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.DigitalOceanVolume is nil")
	}
	return &doVolumeDeleter{
		&doVolume{
			volName: spec.Name(),
			pdName:  spec.PersistentVolume.Spec.DigitalOceanVolume.VolumeID,
			manager: manager,
			plugin:  plugin,
		}}, nil
}

func (plugin *doVolumePlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	return plugin.newProvisionerInternal(options, &DoDiskUtil{})
}

func (plugin *doVolumePlugin) newProvisionerInternal(options volume.VolumeOptions, manager cdManager) (volume.Provisioner, error) {
	return &doVolumeProvisioner{
		doVolume: &doVolume{
			manager: manager,
			plugin:  plugin,
		},
		options: options,
	}, nil
}

func makeGlobalPDName(host volume.VolumeHost, devName string) string {
	return path.Join(host.GetPluginDir(doVolumePluginName), "mounts", devName)
}
