/*
Copyright 2014 The Kubernetes Authors.

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

package qingcloud_volume

import (
	"fmt"
	"os"
	"path"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	kstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&qingcloudVolumePlugin{nil}}
}

type qingcloudVolumePlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &qingcloudVolumePlugin{}
var _ volume.PersistentVolumePlugin = &qingcloudVolumePlugin{}
var _ volume.DeletableVolumePlugin = &qingcloudVolumePlugin{}
var _ volume.ProvisionableVolumePlugin = &qingcloudVolumePlugin{}

const (
	qingcloudVolumePluginName = "kubernetes.io/qingcloud-volume"
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, kstrings.EscapeQualifiedNameForDisk(qingcloudVolumePluginName), volName)
}

func (plugin *qingcloudVolumePlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *qingcloudVolumePlugin) GetPluginName() string {
	return qingcloudVolumePluginName
}

func (plugin *qingcloudVolumePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.VolumeID, nil
}

func (plugin *qingcloudVolumePlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.QingCloudStore != nil) ||
		(spec.Volume != nil && spec.Volume.QingCloudStore != nil)
}

func (plugin *qingcloudVolumePlugin) RequiresRemount() bool {
	return false
}

func (plugin *qingcloudVolumePlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

func (plugin *qingcloudVolumePlugin) NewMounter(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newMounterInternal(spec, pod.UID, &QingDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *qingcloudVolumePlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager volumeManager, mounter mount.Interface) (volume.Mounter, error) {
	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	volumeID := volumeSource.VolumeID
	fsType := volumeSource.FSType

	return &qingcloudVolumeMounter{
		qingcloudVolume: &qingcloudVolume{
			podUID:          podUID,
			volName:         spec.Name(),
			volumeID:        volumeID,
			manager:         manager,
			mounter:         mounter,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, spec.Name(), plugin.host)),
		},
		fsType:   fsType,
		readOnly: readOnly,
	}, nil
}

func (plugin *qingcloudVolumePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newUnmounterInternal(volName, podUID, &QingDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *qingcloudVolumePlugin) newUnmounterInternal(volName string, podUID types.UID, manager volumeManager, mounter mount.Interface) (volume.Unmounter, error) {
	return &qingcloudVolumeUnmounter{&qingcloudVolume{
		podUID:          podUID,
		volName:         volName,
		manager:         manager,
		mounter:         mounter,
		plugin:          plugin,
		MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, volName, plugin.host)),
	}}, nil
}

func (plugin *qingcloudVolumePlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterInternal(spec, &QingDiskUtil{})
}

func (plugin *qingcloudVolumePlugin) newDeleterInternal(spec *volume.Spec, manager volumeManager) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.QingCloudStore == nil {
		glog.Errorf("spec.PersistentVolumeSource.QingCloudStore is nil")
		return nil, fmt.Errorf("spec.PersistentVolumeSource.QingCloudStore is nil")
	}
	return &qingcloudVolumeDeleter{
		qingcloudVolume: &qingcloudVolume{
			volName:  spec.Name(),
			volumeID: spec.PersistentVolume.Spec.QingCloudStore.VolumeID,
			manager:  manager,
			plugin:   plugin,
		}}, nil
}

func (plugin *qingcloudVolumePlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	return plugin.newProvisionerInternal(options, &QingDiskUtil{})
}

func (plugin *qingcloudVolumePlugin) newProvisionerInternal(options volume.VolumeOptions, manager volumeManager) (volume.Provisioner, error) {
	return &qingcloudVolumeProvisioner{
		qingcloudVolume: &qingcloudVolume{
			manager: manager,
			plugin:  plugin,
		},
		options: options,
	}, nil
}

func getVolumeSource(
	spec *volume.Spec) (*api.QingCloudStoreVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.QingCloudStore != nil {
		return spec.Volume.QingCloudStore, spec.Volume.QingCloudStore.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.QingCloudStore != nil {
		return spec.PersistentVolume.Spec.QingCloudStore, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference an qingcloud volume type")
}

func (plugin *qingcloudVolumePlugin) ConstructVolumeSpec(volName, mountPath string) (*volume.Spec, error) {
	mounter := plugin.host.GetMounter()
	pluginDir := plugin.host.GetPluginDir(plugin.GetPluginName())
	sourceName, err := mounter.GetDeviceNameFromMount(mountPath, pluginDir)
	if err != nil {
		return nil, err
	}
	qingVolume := &api.Volume{
		Name: volName,
		VolumeSource: api.VolumeSource{
			QingCloudStore: &api.QingCloudStoreVolumeSource{
				VolumeID: sourceName,
			},
		},
	}
	return volume.NewSpecFromVolume(qingVolume), nil
}

// Abstract interface to PD operations.
type volumeManager interface {
	CreateVolume(provisioner *qingcloudVolumeProvisioner) (volumeID string, volumeSizeGB int, err error)
	DeleteVolume(deleter *qingcloudVolumeDeleter) error
}

// qingcloudVolume are volume resources provided by qingcloud
// that are attached to the kubelet's host machine and exposed to the pod.
type qingcloudVolume struct {
	volName string
	podUID  types.UID
	// Unique id of the PD, used to find the disk resource in the provider.
	volumeID string
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager volumeManager
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	plugin  *qingcloudVolumePlugin
	volume.MetricsProvider
}

func (qcv *qingcloudVolume) GetPath() string {
	return getPath(qcv.podUID, qcv.volName, qcv.plugin.host)
}

type qingcloudVolumeMounter struct {
	*qingcloudVolume
	// Filesystem type, optional.
	fsType string
	// Specifies whether the disk will be mounted as read-only.
	readOnly bool
}

var _ volume.Mounter = &qingcloudVolumeMounter{}

func (b *qingcloudVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: true,
	}
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *qingcloudVolumeMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

// SetUpAt attaches the disk and bind mounts to the volume path.
func (b *qingcloudVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {
	// TODO: handle failed mounts here.
	notMnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	glog.V(4).Infof("PersistentDisk set up: %s %v %v", dir, !notMnt, err)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("cannot validate mount point: %s %v", dir, err)
		return err
	}
	if !notMnt {
		return nil
	}

	globalPDPath := makeGlobalPDPath(b.plugin.host, b.volumeID)

	if err := os.MkdirAll(dir, 0750); err != nil {
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
			glog.Errorf("IsLikelyNotMountPoint check failed for %s: %v", dir, mntErr)
			return err
		}
		if !notMnt {
			if mntErr = b.mounter.Unmount(dir); mntErr != nil {
				glog.Errorf("failed to unmount %s: %v", dir, mntErr)
				return err
			}
			notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("IsLikelyNotMountPoint check failed for %s: %v", dir, mntErr)
				return err
			}
			if !notMnt {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				glog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", dir)
				return err
			}
		}
		os.Remove(dir)
		glog.Errorf("Mount of disk %s failed: %v", dir, err)
		return err
	}

	if !b.readOnly {
		volume.SetVolumeOwnership(b, fsGroup)
	}

	glog.V(4).Infof("Successfully mounted %s", dir)
	return nil
}

func makeGlobalPDPath(host volume.VolumeHost, volumeID string) string {
	return path.Join(host.GetPluginDir(qingcloudVolumePluginName), "mounts", volumeID)
}

type qingcloudVolumeUnmounter struct {
	*qingcloudVolume
}

var _ volume.Unmounter = &qingcloudVolumeUnmounter{}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *qingcloudVolumeUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

// Unmounts the bind mount
func (c *qingcloudVolumeUnmounter) TearDownAt(dir string) error {
	notMnt, err := c.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		glog.V(2).Info("Error checking if mountpoint ", dir, ": ", err)
		return err
	}
	if notMnt {
		glog.V(2).Info("Not mountpoint, deleting")
		return os.Remove(dir)
	}

	// Unmount the bind-mount inside this pod
	if err := c.mounter.Unmount(dir); err != nil {
		glog.V(2).Info("Error unmounting dir ", dir, ": ", err)
		return err
	}
	notMnt, mntErr := c.mounter.IsLikelyNotMountPoint(dir)
	if mntErr != nil {
		glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
		return err
	}
	if notMnt {
		if err := os.Remove(dir); err != nil {
			glog.V(2).Info("Error removing mountpoint ", dir, ": ", err)
			return err
		}
	}
	return nil
}

type qingcloudVolumeDeleter struct {
	*qingcloudVolume
}

var _ volume.Deleter = &qingcloudVolumeDeleter{}

func (d *qingcloudVolumeDeleter) Delete() error {
	return d.manager.DeleteVolume(d)
}

type qingcloudVolumeProvisioner struct {
	*qingcloudVolume
	options volume.VolumeOptions
}

var _ volume.Provisioner = &qingcloudVolumeProvisioner{}

func (c *qingcloudVolumeProvisioner) Provision() (*api.PersistentVolume, error) {
	volumeID, sizeGB, err := c.manager.CreateVolume(c)
	if err != nil {
		glog.Errorf("Provision failed: %v", err)
		return nil, err
	}

	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: c.options.PVName,
			Annotations: map[string]string{
				"kubernetes.io/createdby": "qingcloud-volume-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: c.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   c.options.PVC.Spec.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				QingCloudStore: &api.QingCloudStoreVolumeSource{
					VolumeID: volumeID,
					FSType:   "ext4",
					ReadOnly: false,
				},
			},
		},
	}

	if len(c.options.PVC.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = c.plugin.GetAccessModes()
	}

	return pv, nil
}
