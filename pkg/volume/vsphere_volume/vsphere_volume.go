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
	"fmt"
	"os"
	"path"
	"strings"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
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

func (plugin *vsphereVolumePlugin) SupportsMountOption() bool {
	return true
}

func (plugin *vsphereVolumePlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *vsphereVolumePlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
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

func (plugin *vsphereVolumePlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	mounter := plugin.host.GetMounter()
	pluginDir := plugin.host.GetPluginDir(plugin.GetPluginName())
	volumePath, err := mounter.GetDeviceNameFromMount(mountPath, pluginDir)
	if err != nil {
		return nil, err
	}
	volumePath = strings.Replace(volumePath, "\\040", " ", -1)
	glog.V(5).Infof("vSphere volume path is %q", volumePath)
	vsphereVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
				VolumePath: volumePath,
			},
		},
	}
	return volume.NewSpecFromVolume(vsphereVolume), nil
}

// Abstract interface to disk operations.
type vdManager interface {
	// Creates a volume
	CreateVolume(provisioner *vsphereVolumeProvisioner) (volSpec *VolumeSpec, err error)
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

var _ volume.Mounter = &vsphereVolumeMounter{}

type vsphereVolumeMounter struct {
	*vsphereVolume
	fsType      string
	diskMounter *mount.SafeFormatAndMount
}

func (b *vsphereVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		SupportsSELinux: true,
		Managed:         true,
	}
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *vsphereVolumeMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *vsphereVolumeMounter) CanMount() error {
	return nil
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

	if err := os.MkdirAll(dir, 0750); err != nil {
		glog.V(4).Infof("Could not create directory %s: %v", dir, err)
		return err
	}

	options := []string{"bind"}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	globalPDPath := makeGlobalPDPath(b.plugin.host, b.volPath)
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
		return err
	}
	volume.SetVolumeOwnership(b, fsGroup)
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
	return util.UnmountPath(dir, v.mounter)
}

func makeGlobalPDPath(host volume.VolumeHost, devName string) string {
	return path.Join(host.GetPluginDir(vsphereVolumePluginName), mount.MountsInGlobalPDPath, devName)
}

func (vv *vsphereVolume) GetPath() string {
	name := vsphereVolumePluginName
	return vv.plugin.host.GetPodVolumeDir(vv.podUID, utilstrings.EscapeQualifiedNameForDisk(name), vv.volName)
}

// vSphere Persistent Volume Plugin
func (plugin *vsphereVolumePlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
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

func (v *vsphereVolumeProvisioner) Provision() (*v1.PersistentVolume, error) {
	if !volume.AccessModesContainedInAll(v.plugin.GetAccessModes(), v.options.PVC.Spec.AccessModes) {
		return nil, fmt.Errorf("invalid AccessModes %v: only AccessModes %v are supported", v.options.PVC.Spec.AccessModes, v.plugin.GetAccessModes())
	}

	volSpec, err := v.manager.CreateVolume(v)
	if err != nil {
		return nil, err
	}

	if volSpec.Fstype == "" {
		volSpec.Fstype = "ext4"
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:   v.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				volumehelper.VolumeDynamicallyCreatedByKey: "vsphere-volume-dynamic-provisioner",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: v.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   v.options.PVC.Spec.AccessModes,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dKi", volSpec.Size)),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
					VolumePath:        volSpec.Path,
					FSType:            volSpec.Fstype,
					StoragePolicyName: volSpec.StoragePolicyName,
					StoragePolicyID:   volSpec.StoragePolicyID,
				},
			},
		},
	}
	if len(v.options.PVC.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = v.plugin.GetAccessModes()
	}

	return pv, nil
}

func getVolumeSource(
	spec *volume.Spec) (*v1.VsphereVirtualDiskVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.VsphereVolume != nil {
		return spec.Volume.VsphereVolume, spec.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.VsphereVolume != nil {
		return spec.PersistentVolume.Spec.VsphereVolume, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a VSphere volume type")
}
