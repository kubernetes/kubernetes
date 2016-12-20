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

package photon_pd

import (
	"fmt"
	"os"
	"path"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&photonPersistentDiskPlugin{}}
}

type photonPersistentDiskPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &photonPersistentDiskPlugin{}
var _ volume.PersistentVolumePlugin = &photonPersistentDiskPlugin{}
var _ volume.DeletableVolumePlugin = &photonPersistentDiskPlugin{}
var _ volume.ProvisionableVolumePlugin = &photonPersistentDiskPlugin{}

const (
	photonPersistentDiskPluginName = "kubernetes.io/photon-pd"
)

func (plugin *photonPersistentDiskPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *photonPersistentDiskPlugin) GetPluginName() string {
	return photonPersistentDiskPluginName
}

func (plugin *photonPersistentDiskPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		glog.Errorf("Photon volume plugin: GetVolumeName failed to get volume source")
		return "", err
	}

	return volumeSource.PdID, nil
}

func (plugin *photonPersistentDiskPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.PhotonPersistentDisk != nil) ||
		(spec.Volume != nil && spec.Volume.PhotonPersistentDisk != nil)
}

func (plugin *photonPersistentDiskPlugin) RequiresRemount() bool {
	return false
}

func (plugin *photonPersistentDiskPlugin) NewMounter(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod.UID, &PhotonDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *photonPersistentDiskPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, &PhotonDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *photonPersistentDiskPlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager pdManager, mounter mount.Interface) (volume.Mounter, error) {
	vvol, _, err := getVolumeSource(spec)
	if err != nil {
		glog.Errorf("Photon volume plugin: newMounterInternal failed to get volume source")
		return nil, err
	}

	pdID := vvol.PdID
	fsType := vvol.FSType

	return &photonPersistentDiskMounter{
		photonPersistentDisk: &photonPersistentDisk{
			podUID:  podUID,
			volName: spec.Name(),
			pdID:    pdID,
			manager: manager,
			mounter: mounter,
			plugin:  plugin,
		},
		fsType:      fsType,
		diskMounter: &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}}, nil
}

func (plugin *photonPersistentDiskPlugin) newUnmounterInternal(volName string, podUID types.UID, manager pdManager, mounter mount.Interface) (volume.Unmounter, error) {
	return &photonPersistentDiskUnmounter{
		&photonPersistentDisk{
			podUID:  podUID,
			volName: volName,
			manager: manager,
			mounter: mounter,
			plugin:  plugin,
		}}, nil
}

func (plugin *photonPersistentDiskPlugin) ConstructVolumeSpec(volumeSpecName, mountPath string) (*volume.Spec, error) {
	mounter := plugin.host.GetMounter()
	pluginDir := plugin.host.GetPluginDir(plugin.GetPluginName())
	pdID, err := mounter.GetDeviceNameFromMount(mountPath, pluginDir)
	if err != nil {
		return nil, err
	}

	photonPersistentDisk := &api.Volume{
		Name: volumeSpecName,
		VolumeSource: api.VolumeSource{
			PhotonPersistentDisk: &api.PhotonPersistentDiskVolumeSource{
				PdID: pdID,
			},
		},
	}
	return volume.NewSpecFromVolume(photonPersistentDisk), nil
}

// Abstract interface to disk operations.
type pdManager interface {
	// Creates a volume
	CreateVolume(provisioner *photonPersistentDiskProvisioner) (pdID string, volumeSizeGB int, err error)
	// Deletes a volume
	DeleteVolume(deleter *photonPersistentDiskDeleter) error
}

// photonPersistentDisk volumes are disk resources are attached to the kubelet's host machine and exposed to the pod.
type photonPersistentDisk struct {
	volName string
	podUID  types.UID
	// Unique identifier of the volume, used to find the disk resource in the provider.
	pdID string
	// Filesystem type, optional.
	fsType string
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager pdManager
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	plugin  *photonPersistentDiskPlugin
	volume.MetricsNil
}

var _ volume.Mounter = &photonPersistentDiskMounter{}

type photonPersistentDiskMounter struct {
	*photonPersistentDisk
	fsType      string
	diskMounter *mount.SafeFormatAndMount
}

func (b *photonPersistentDiskMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		SupportsSELinux: true,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *photonPersistentDiskMounter) CanMount() error {
	return nil
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *photonPersistentDiskMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *photonPersistentDiskMounter) SetUpAt(dir string, fsGroup *int64) error {
	glog.V(4).Infof("Photon Persistent Disk setup %s to %s", b.pdID, dir)

	// TODO: handle failed mounts here.
	notmnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("cannot validate mount point: %s %v", dir, err)
		return err
	}
	if !notmnt {
		return nil
	}

	if err := os.MkdirAll(dir, 0750); err != nil {
		glog.Errorf("mkdir failed on disk %s (%v)", dir, err)
		return err
	}

	options := []string{"bind"}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	globalPDPath := makeGlobalPDPath(b.plugin.host, b.pdID)
	glog.V(4).Infof("attempting to mount %s", dir)

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
		glog.Errorf("Mount of disk %s failed: %v", dir, err)
		return err
	}

	return nil
}

var _ volume.Unmounter = &photonPersistentDiskUnmounter{}

type photonPersistentDiskUnmounter struct {
	*photonPersistentDisk
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *photonPersistentDiskUnmounter) TearDown() error {
	err := c.TearDownAt(c.GetPath())
	if err != nil {
		return err
	}

	removeFromScsiSubsystem(c.volName)
	return nil
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *photonPersistentDiskUnmounter) TearDownAt(dir string) error {
	return util.UnmountPath(dir, c.mounter)
}

func makeGlobalPDPath(host volume.VolumeHost, devName string) string {
	return path.Join(host.GetPluginDir(photonPersistentDiskPluginName), mount.MountsInGlobalPDPath, devName)
}

func (ppd *photonPersistentDisk) GetPath() string {
	name := photonPersistentDiskPluginName
	return ppd.plugin.host.GetPodVolumeDir(ppd.podUID, utilstrings.EscapeQualifiedNameForDisk(name), ppd.volName)
}

// TODO: supporting more access mode for PhotonController persistent disk
func (plugin *photonPersistentDiskPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

type photonPersistentDiskDeleter struct {
	*photonPersistentDisk
}

var _ volume.Deleter = &photonPersistentDiskDeleter{}

func (plugin *photonPersistentDiskPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterInternal(spec, &PhotonDiskUtil{})
}

func (plugin *photonPersistentDiskPlugin) newDeleterInternal(spec *volume.Spec, manager pdManager) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.PhotonPersistentDisk == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.PhotonPersistentDisk is nil")
	}
	return &photonPersistentDiskDeleter{
		&photonPersistentDisk{
			volName: spec.Name(),
			pdID:    spec.PersistentVolume.Spec.PhotonPersistentDisk.PdID,
			manager: manager,
			plugin:  plugin,
		}}, nil
}

func (r *photonPersistentDiskDeleter) Delete() error {
	return r.manager.DeleteVolume(r)
}

type photonPersistentDiskProvisioner struct {
	*photonPersistentDisk
	options volume.VolumeOptions
}

var _ volume.Provisioner = &photonPersistentDiskProvisioner{}

func (plugin *photonPersistentDiskPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	return plugin.newProvisionerInternal(options, &PhotonDiskUtil{})
}

func (plugin *photonPersistentDiskPlugin) newProvisionerInternal(options volume.VolumeOptions, manager pdManager) (volume.Provisioner, error) {
	return &photonPersistentDiskProvisioner{
		photonPersistentDisk: &photonPersistentDisk{
			manager: manager,
			plugin:  plugin,
		},
		options: options,
	}, nil
}

func (p *photonPersistentDiskProvisioner) Provision() (*api.PersistentVolume, error) {
	pdID, sizeGB, err := p.manager.CreateVolume(p)
	if err != nil {
		return nil, err
	}

	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:   p.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "photon-volume-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: p.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   p.options.PVC.Spec.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				PhotonPersistentDisk: &api.PhotonPersistentDiskVolumeSource{
					PdID:   pdID,
					FSType: "ext4",
				},
			},
		},
	}
	if len(p.options.PVC.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = p.plugin.GetAccessModes()
	}

	return pv, nil
}

func getVolumeSource(
	spec *volume.Spec) (*api.PhotonPersistentDiskVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.PhotonPersistentDisk != nil {
		return spec.Volume.PhotonPersistentDisk, spec.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.PhotonPersistentDisk != nil {
		return spec.PersistentVolume.Spec.PhotonPersistentDisk, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a Photon Controller persistent disk type")
}
