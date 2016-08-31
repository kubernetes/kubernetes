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

package cinder_local

import (
	"fmt"
	"os"
	"path"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&cinderLocalPlugin{}}
}

type cinderLocalPlugin struct {
	host volume.VolumeHost
	// Guarding SetUp and TearDown operations
	volumeLocks keymutex.KeyMutex
}

var _ volume.VolumePlugin = &cinderLocalPlugin{}
var _ volume.PersistentVolumePlugin = &cinderLocalPlugin{}
var _ volume.DeletableVolumePlugin = &cinderLocalPlugin{}
var _ volume.ProvisionableVolumePlugin = &cinderLocalPlugin{}

const (
	cinderLocalVolumePluginName = "kubernetes.io/cinder-local"
)

func (plugin *cinderLocalPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	plugin.volumeLocks = keymutex.NewKeyMutex()
	return nil
}

func (plugin *cinderLocalPlugin) GetPluginName() string {
	return cinderLocalVolumePluginName
}

func (plugin *cinderLocalPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.VolumeID, nil
}

func (plugin *cinderLocalPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.Volume != nil && spec.Volume.CinderLocal != nil) || (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.CinderLocal != nil)
}

func (plugin *cinderLocalPlugin) RequiresRemount() bool {
	return false
}

func (plugin *cinderLocalPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

func (plugin *cinderLocalPlugin) NewMounter(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod, &cinderDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *cinderLocalPlugin) newMounterInternal(spec *volume.Spec, pod *api.Pod, manager cdManager, mounter mount.Interface) (volume.Mounter, error) {
	source, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	return &cinderVolumeMounter{
		cinderVolume: &cinderVolume{
			volName: spec.Name(),
			mounter: mounter,
			manager: manager,
			plugin:  plugin,
		},
		pod:      pod,
		source:   source,
		readOnly: readOnly,
	}, nil
}

func (plugin *cinderLocalPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, &cinderDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *cinderLocalPlugin) newUnmounterInternal(volName string, podUID types.UID, manager cdManager, mounter mount.Interface) (volume.Unmounter, error) {
	return &cinderVolumeUnmounter{
		cinderVolume: &cinderVolume{
			volName: volName,
			manager: manager,
			mounter: mounter,
			plugin:  plugin,
		},
		podUID: podUID,
	}, nil
}

func (plugin *cinderLocalPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterInternal(spec, &cinderDiskUtil{})
}

func (plugin *cinderLocalPlugin) newDeleterInternal(spec *volume.Spec, manager cdManager) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.CinderLocal == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.CinderLocal is nil")
	}

	return &cinderVolumeDeleter{
		cinderVolume: &cinderVolume{
			volName: spec.Name(),
			manager: manager,
			plugin:  plugin,
		},
		pv: spec.PersistentVolume,
	}, nil
}

func (plugin *cinderLocalPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	if len(options.AccessModes) == 0 {
		options.AccessModes = plugin.GetAccessModes()
	}
	return plugin.newProvisionerInternal(options, &cinderDiskUtil{})
}

func (plugin *cinderLocalPlugin) newProvisionerInternal(options volume.VolumeOptions, manager cdManager) (volume.Provisioner, error) {
	return &cinderVolumeProvisioner{
		cinderVolume: &cinderVolume{
			manager: manager,
			plugin:  plugin,
		},
		options: options,
	}, nil
}

func (plugin *cinderLocalPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	mounter := plugin.host.GetMounter()
	pluginDir := plugin.host.GetPluginDir(plugin.GetPluginName())
	sourceName, err := mounter.GetDeviceNameFromMount(mountPath, pluginDir)
	if err != nil {
		return nil, fmt.Errorf("could not get device name from mount point: %v", err)
	}
	glog.V(4).Infof("Found volume %s mounted to %s", sourceName, mountPath)
	cinderVolume := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			CinderLocal: &api.CinderLocalVolumeSource{
				CinderVolumeSource: api.CinderVolumeSource{
					VolumeID: sourceName,
					FSType:   "",
					ReadOnly: false,
				},
			},
		},
	}
	return volume.NewSpecFromVolume(cinderVolume), nil
}

// Abstract interface to PD operations.
type cdManager interface {
	// Attaches the disk to the kubelet's host machine.
	AttachDisk(mounter *cinderVolumeMounter, mntPoint string) error
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(unmounter *cinderVolumeUnmounter, mntPoint string) error
	// Creates a volume
	CreateVolume(provisioner *cinderVolumeProvisioner) (volumeID string, volumeSizeGB int, secretRef string, err error)
	// Deletes a volume
	DeleteVolume(deleter *cinderVolumeDeleter) error
}

// cinderPersistentDisk volumes are disk resources provided by C3
// that are attached to the kubelet's host machine and exposed to the pod.
type cinderVolume struct {
	volName string
	// Specifies the partition to mount
	//partition string
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager cdManager
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	plugin  *cinderLocalPlugin
	volume.MetricsNil
}

type cinderVolumeMounter struct {
	*cinderVolume
	readOnly bool
	pod      *api.Pod
	// Specifies the cinder volume details
	source *api.CinderLocalVolumeSource
}

var _ volume.Mounter = &cinderVolumeMounter{}

func detachDiskLogError(cd *cinderVolume, podUID types.UID, mntPoint string) {
	u := cinderVolumeUnmounter{
		cinderVolume: cd,
		podUID:       podUID,
	}
	if err := cd.manager.DetachDisk(&u, mntPoint); err != nil {
		glog.Warningf("Failed to detach disk: %v (%v)", cd, err)
	}
}

func (m *cinderVolumeMounter) GetPath() string {
	return m.getPath(m.pod.UID)
}

func (m *cinderVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        m.readOnly,
		Managed:         !m.readOnly,
		SupportsSELinux: true,
	}
}

func (m *cinderVolumeMounter) SetUp(fsGroup *int64) error {
	return m.SetUpAt(m.GetPath(), fsGroup)
}

// SetUp bind mounts to the volume path.
func (m *cinderVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {
	m.plugin.volumeLocks.LockKey(m.source.VolumeID)
	defer m.plugin.volumeLocks.UnlockKey(m.source.VolumeID)

	globalPDPath := makeGlobalPDName(m.plugin.host, m.source.VolumeID)

	// TODO: handle failed mounts here.
	notmnt, err := m.mounter.IsLikelyNotMountPoint(dir)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("cannot validate mount point: %s %v", dir, err)
	}
	if !notmnt {
		glog.V(4).Infof("Something is already mounted to target %s", dir)
		return nil
	}

	if err := m.manager.AttachDisk(m, globalPDPath); err != nil {
		return err
	}

	options := []string{"bind"}
	if m.readOnly {
		options = append(options, "ro")
	}

	if err := os.MkdirAll(dir, 0750); err != nil {
		// TODO: we should really eject the attach/detach out into its own control loop.
		glog.V(4).Infof("Could not create directory %s: %v", dir, err)
		detachDiskLogError(m.cinderVolume, m.pod.UID, globalPDPath)
		return fmt.Errorf("could not create %s: %v", dir, err)
	}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	glog.V(4).Infof("Attempting to mount cinder volume %s to %s with options %v", m.source.VolumeID, dir, options)
	err = m.mounter.Mount(globalPDPath, dir, "", options)
	if err != nil {
		return fmt.Errorf("failed to mount %s: %v", dir, err)
	}

	if !m.readOnly {
		volume.SetVolumeOwnership(m, fsGroup)
	}
	glog.V(3).Infof("Cinder volume %s mounted to %s", m.source.VolumeID, dir)

	return nil
}

func makeGlobalPDName(host volume.VolumeHost, devName string) string {
	return path.Join(host.GetPluginDir(cinderLocalVolumePluginName), "mounts", devName)
}

func (cd *cinderVolume) getPath(podUID types.UID) string {
	name := cinderLocalVolumePluginName
	return cd.plugin.host.GetPodVolumeDir(podUID, strings.EscapeQualifiedNameForDisk(name), cd.volName)
}

type cinderVolumeUnmounter struct {
	*cinderVolume
	podUID   types.UID
	volumeID string
}

var _ volume.Unmounter = &cinderVolumeUnmounter{}

func (u *cinderVolumeUnmounter) GetPath() string {
	return u.getPath(u.podUID)
}

func (u *cinderVolumeUnmounter) TearDown() error {
	return u.TearDownAt(u.GetPath())
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (u *cinderVolumeUnmounter) TearDownAt(dir string) error {
	glog.V(5).Infof("Cinder TearDown of %s", dir)

	// lock the volume (and thus wait for any concurrrent SetUpAt to finish)
	u.plugin.volumeLocks.LockKey(u.volumeID)
	defer u.plugin.volumeLocks.UnlockKey(u.volumeID)

	refs, err := mount.GetMountRefs(u.mounter, dir)
	if err != nil {
		glog.V(4).Infof("GetMountRefs failed: %v", err)
		return fmt.Errorf("could not get other mountpoints for %s: %v", dir, err)
	}
	if len(refs) > 0 {
		if err = u.mounter.Unmount(dir); err != nil {
			glog.V(4).Infof("Unmount failed: %v", err)
			return fmt.Errorf("failed to unmount %s: %v", dir, err)
		}
		glog.V(3).Infof("Successfully unmounted: %s\n", dir)

		if len(refs) == 1 {
			// this was the last bind mount
			u.volumeID = path.Base(refs[0])
			globalPDPath := makeGlobalPDName(u.plugin.host, u.volumeID)
			if err = u.manager.DetachDisk(u, globalPDPath); err != nil {
				return err
			}
		}
	}

	if err = os.RemoveAll(dir); err != nil {
		return fmt.Errorf("failed to remove mountpoint %s: %v", dir, err)
	}
	return nil
}

type cinderVolumeDeleter struct {
	*cinderVolume
	pv *api.PersistentVolume
}

var _ volume.Deleter = &cinderVolumeDeleter{}

func (d *cinderVolumeDeleter) GetPath() string {
	// GetPath is part of Deleter interface but I think it's by mistake
	glog.V(4).Info("cinderVolumeDeleter.GetPath() called but not implemented")
	return ""
}

func (d *cinderVolumeDeleter) Delete() error {
	if err := d.manager.DeleteVolume(d); err != nil {
		return err
	}

	glog.V(2).Infof("CinderLocal volume %v deleted successfuly", d.volName)
	return nil
}

type cinderVolumeProvisioner struct {
	*cinderVolume
	options volume.VolumeOptions
}

var _ volume.Provisioner = &cinderVolumeProvisioner{}

func (p *cinderVolumeProvisioner) Provision() (*api.PersistentVolume, error) {
	volumeID, sizeGB, secretRef, err := p.manager.CreateVolume(p)
	if err != nil {
		return nil, err
	}

	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:   p.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "cinder-local-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: p.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   p.options.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				CinderLocal: &api.CinderLocalVolumeSource{
					CinderVolumeSource: api.CinderVolumeSource{
						VolumeID: volumeID,
						FSType:   "ext4",
						ReadOnly: false,
					},
					SecretRef: secretRef,
				},
			},
		},
	}

	glog.V(2).Infof("CinderLocal volume %v provisioned successfuly", volumeID)
	return pv, nil
}

func getVolumeSource(spec *volume.Spec) (*api.CinderLocalVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.CinderLocal != nil {
		return spec.Volume.CinderLocal, spec.Volume.CinderLocal.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.CinderLocal != nil {
		return spec.PersistentVolume.Spec.CinderLocal, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a CinderLocal volume type")
}
