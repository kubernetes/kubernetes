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

package vsphere_volume

import (
	"errors"
	"fmt"
	"os"
	"path"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&vsphereVolumePlugin{}}
}

type vsphereVolumePlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &vsphereVolumePlugin{}
var _ volume.PersistentVolumePlugin = &vsphereVolumePlugin{}
var _ volume.DeletableVolumePlugin = &vsphereVolumePlugin{}
var _ volume.ProvisionableVolumePlugin = &vsphereVolumePlugin{}

const (
	vsphereVolumePluginName = "kubernetes.io/vsphere-volume"
)

// vSphere Volume Plugin
func (plugin *vsphereVolumePlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *vsphereVolumePlugin) GetPluginName() string {
	return vsphereVolumePluginName
}

func (plugin *vsphereVolumePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.VolumePath, nil
}

func (plugin *vsphereVolumePlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.VsphereVolume != nil) ||
		(spec.Volume != nil && spec.Volume.VsphereVolume != nil)
}

func (plugin *vsphereVolumePlugin) RequiresRemount() bool {
	return false
}

func (plugin *vsphereVolumePlugin) NewMounter(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod.UID, &VsphereDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *vsphereVolumePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, &VsphereDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *vsphereVolumePlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager vdManager, mounter mount.Interface) (volume.Mounter, error) {
	vvol, _, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	volPath := vvol.VolumePath
	fsType := vvol.FSType

	return &vsphereVolumeMounter{
		vsphereVolume: &vsphereVolume{
			podUID:  podUID,
			volName: spec.Name(),
			volPath: volPath,
			manager: manager,
			mounter: mounter,
			plugin:  plugin,
		},
		fsType:      fsType,
		diskMounter: &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}}, nil
}

func (plugin *vsphereVolumePlugin) newUnmounterInternal(volName string, podUID types.UID, manager vdManager, mounter mount.Interface) (volume.Unmounter, error) {
	return &vsphereVolumeUnmounter{
		&vsphereVolume{
			podUID:  podUID,
			volName: volName,
			manager: manager,
			mounter: mounter,
			plugin:  plugin,
		}}, nil
}

func (plugin *vsphereVolumePlugin) getCloudProvider() (*vsphere.VSphere, error) {
	cloud := plugin.host.GetCloudProvider()
	if cloud == nil {
		glog.Errorf("Cloud provider not initialized properly")
		return nil, errors.New("Cloud provider not initialized properly")
	}

	vs := cloud.(*vsphere.VSphere)
	if vs == nil {
		return nil, errors.New("Invalid cloud provider: expected vSphere")
	}
	return vs, nil
}

func (plugin *vsphereVolumePlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	vsphereVolume := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			VsphereVolume: &api.VsphereVirtualDiskVolumeSource{
				VolumePath: volumeName,
			},
		},
	}
	return volume.NewSpecFromVolume(vsphereVolume), nil
}

// Abstract interface to disk operations.
type vdManager interface {
	// Attaches the disk to the kubelet's host machine.
	AttachDisk(mounter *vsphereVolumeMounter, globalPDPath string) error
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(unmounter *vsphereVolumeUnmounter) error
	// Creates a volume
	CreateVolume(provisioner *vsphereVolumeProvisioner) (vmDiskPath string, volumeSizeGB int, err error)
	// Deletes a volume
	DeleteVolume(deleter *vsphereVolumeDeleter) error
}

// vspherePersistentDisk volumes are disk resources are attached to the kubelet's host machine and exposed to the pod.
type vsphereVolume struct {
	volName string
	podUID  types.UID
	// Unique identifier of the volume, used to find the disk resource in the provider.
	volPath string
	// Filesystem type, optional.
	fsType string
	//diskID for detach disk
	diskID string
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager vdManager
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	// diskMounter provides the interface that is used to mount the actual block device.
	diskMounter mount.Interface
	plugin      *vsphereVolumePlugin
	volume.MetricsNil
}

func detachDiskLogError(vv *vsphereVolume) {
	err := vv.manager.DetachDisk(&vsphereVolumeUnmounter{vv})
	if err != nil {
		glog.Warningf("Failed to detach disk: %v (%v)", vv, err)
	}
}

var _ volume.Mounter = &vsphereVolumeMounter{}

type vsphereVolumeMounter struct {
	*vsphereVolume
	fsType      string
	diskMounter *mount.SafeFormatAndMount
}

func (b *vsphereVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		SupportsSELinux: true,
	}
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *vsphereVolumeMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *vsphereVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {
	glog.V(5).Infof("vSphere volume setup %s to %s", b.volPath, dir)

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
	globalPDPath := makeGlobalPDPath(b.plugin.host, b.volPath)
	if err := b.manager.AttachDisk(b, globalPDPath); err != nil {
		glog.V(3).Infof("AttachDisk failed: %v", err)
		return err
	}
	glog.V(3).Infof("vSphere volume %s attached", b.volPath)

	options := []string{"bind"}

	if err := os.MkdirAll(dir, 0750); err != nil {
		// TODO: we should really eject the attach/detach out into its own control loop.
		glog.V(4).Infof("Could not create directory %s: %v", dir, err)
		detachDiskLogError(b.vsphereVolume)
		return err
	}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	err = b.mounter.Mount(globalPDPath, dir, "", options)
	if err != nil {
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
				glog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", b.GetPath())
				return err
			}
		}
		os.Remove(dir)
		detachDiskLogError(b.vsphereVolume)
		return err
	}
	glog.V(3).Infof("vSphere volume %s mounted to %s", b.volPath, dir)

	return nil
}

var _ volume.Unmounter = &vsphereVolumeUnmounter{}

type vsphereVolumeUnmounter struct {
	*vsphereVolume
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (v *vsphereVolumeUnmounter) TearDown() error {
	return v.TearDownAt(v.GetPath())
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (v *vsphereVolumeUnmounter) TearDownAt(dir string) error {
	glog.V(5).Infof("vSphere Volume TearDown of %s", dir)
	notmnt, err := v.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		glog.V(4).Infof("Error checking if mountpoint ", dir, ": ", err)
		return err
	}
	if notmnt {
		glog.V(4).Infof("Not mount point,deleting")
		return os.Remove(dir)
	}

	// Find vSphere volumeID to lock the right volume
	refs, err := mount.GetMountRefs(v.mounter, dir)
	if err != nil {
		glog.V(4).Infof("Error getting mountrefs for ", dir, ": ", err)
		return err
	}
	if len(refs) == 0 {
		glog.V(4).Infof("Directory %s is not mounted", dir)
		return fmt.Errorf("directory %s is not mounted", dir)
	}

	mountPath := refs[0]
	// Assumption: No file or folder is named starting with '[' in datastore
	volumePath := mountPath[strings.LastIndex(mountPath, "["):]
	// space between datastore and vmdk name in volumePath is encoded as '\040' when returned by GetMountRefs().
	// volumePath eg: "[local] xxx.vmdk" provided to attach/mount
	// replacing \040 with space to match the actual volumePath
	v.volPath = strings.Replace(volumePath, "\\040", " ", -1)
	glog.V(4).Infof("Found volume %s mounted to %s", v.volPath, dir)

	// Reload list of references, there might be SetUpAt finished in the meantime
	refs, err = mount.GetMountRefs(v.mounter, dir)
	if err != nil {
		glog.V(4).Infof("GetMountRefs failed: %v", err)
		return err
	}
	if err := v.mounter.Unmount(dir); err != nil {
		glog.V(4).Infof("Unmount failed: %v", err)
		return err
	}
	glog.V(3).Infof("Successfully unmounted: %s\n", dir)

	// If refCount is 1, then all bind mounts have been removed, and the
	// remaining reference is the global mount. It is safe to detach.
	if len(refs) == 1 {
		if err := v.manager.DetachDisk(v); err != nil {
			glog.V(4).Infof("DetachDisk failed: %v", err)
			return err
		}
		glog.V(3).Infof("Volume %s detached", v.volPath)
	}
	notmnt, mntErr := v.mounter.IsLikelyNotMountPoint(dir)
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

func makeGlobalPDPath(host volume.VolumeHost, devName string) string {
	return path.Join(host.GetPluginDir(vsphereVolumePluginName), "mounts", devName)
}

func (vv *vsphereVolume) GetPath() string {
	name := vsphereVolumePluginName
	return vv.plugin.host.GetPodVolumeDir(vv.podUID, utilstrings.EscapeQualifiedNameForDisk(name), vv.volName)
}

// vSphere Persistent Volume Plugin
func (plugin *vsphereVolumePlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

// vSphere Deletable Volume Plugin
type vsphereVolumeDeleter struct {
	*vsphereVolume
}

var _ volume.Deleter = &vsphereVolumeDeleter{}

func (plugin *vsphereVolumePlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterInternal(spec, &VsphereDiskUtil{})
}

func (plugin *vsphereVolumePlugin) newDeleterInternal(spec *volume.Spec, manager vdManager) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.VsphereVolume == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.VsphereVolume is nil")
	}
	return &vsphereVolumeDeleter{
		&vsphereVolume{
			volName: spec.Name(),
			volPath: spec.PersistentVolume.Spec.VsphereVolume.VolumePath,
			manager: manager,
			plugin:  plugin,
		}}, nil
}

func (r *vsphereVolumeDeleter) Delete() error {
	return r.manager.DeleteVolume(r)
}

// vSphere Provisionable Volume Plugin
type vsphereVolumeProvisioner struct {
	*vsphereVolume
	options volume.VolumeOptions
}

var _ volume.Provisioner = &vsphereVolumeProvisioner{}

func (plugin *vsphereVolumePlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	if len(options.AccessModes) == 0 {
		options.AccessModes = plugin.GetAccessModes()
	}
	return plugin.newProvisionerInternal(options, &VsphereDiskUtil{})
}

func (plugin *vsphereVolumePlugin) newProvisionerInternal(options volume.VolumeOptions, manager vdManager) (volume.Provisioner, error) {
	return &vsphereVolumeProvisioner{
		vsphereVolume: &vsphereVolume{
			manager: manager,
			plugin:  plugin,
		},
		options: options,
	}, nil
}

func (v *vsphereVolumeProvisioner) Provision() (*api.PersistentVolume, error) {
	vmDiskPath, sizeKB, err := v.manager.CreateVolume(v)
	if err != nil {
		return nil, err
	}

	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:   v.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "vsphere-volume-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: v.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   v.options.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dKi", sizeKB)),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				VsphereVolume: &api.VsphereVirtualDiskVolumeSource{
					VolumePath: vmDiskPath,
					FSType:     "ext4",
				},
			},
		},
	}
	return pv, nil
}

func getVolumeSource(
	spec *volume.Spec) (*api.VsphereVirtualDiskVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.VsphereVolume != nil {
		return spec.Volume.VsphereVolume, spec.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.VsphereVolume != nil {
		return spec.PersistentVolume.Spec.VsphereVolume, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a VSphere volume type")
}
