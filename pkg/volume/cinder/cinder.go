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

package cinder

import (
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	kstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

const cinderVolumePluginName = "kubernetes.io/cinder"

var _ volume.VolumePlugin = &cinderPlugin{}
var _ volume.PersistentVolumePlugin = &cinderPlugin{}
var _ volume.DeletableVolumePlugin = &cinderPlugin{}
var _ volume.ProvisionableVolumePlugin = &cinderPlugin{}
var _ volume.AttachableVolumePlugin = &cinderPlugin{}

var _ volume.Mounter = &cinderVolumeMounter{}
var _ volume.Unmounter = &cinderVolumeUnmounter{}
var _ volume.Deleter = &cinderVolumeDeleter{}
var _ volume.Provisioner = &cinderVolumeProvisioner{}
var _ volume.Attacher = &cinderDiskAttacher{}
var _ volume.Detacher = &cinderDiskDetacher{}

type cinderVolume struct {
	volName string
	podUID  types.UID
	// Unique identifier of the volume, used to find the disk resource in the provider.
	pdName string
	// Filesystem type, optional.
	fsType string
	// Specifies the partition to mount
	// partition string
	// Specifies whether the disk will be attached as read-only.
	readOnly bool
	// Secret that contains the keystone credentials, optional.
	secretRef *v1.LocalObjectReference
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager cinderManager
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	// diskMounter provides the interface that is used to mount the actual block device.
	blockDeviceMounter mount.Interface
	plugin             *cinderPlugin
	volume.MetricsNil
}

func (cd *cinderVolume) GetPath() string {
	name := cinderVolumePluginName
	return cd.plugin.host.GetPodVolumeDir(cd.podUID, kstrings.EscapeQualifiedNameForDisk(name), cd.volName)
}

type cinderPlugin struct {
	host volume.VolumeHost
	// Guarding SetUp and TearDown operations
	volumeLocks keymutex.KeyMutex
	// For tests purposes, otherwise detected using host:
	cinderProvider CinderProvider
}

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&cinderPlugin{}}
}

// VolumePlugin interface.

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

func (plugin *cinderPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	return newMounter(spec, pod.UID, plugin, plugin.host.GetMounter())
}

func (plugin *cinderPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return newUnmounter(volName, podUID, plugin, plugin.host.GetMounter())
}

func (plugin *cinderPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	mounter := plugin.host.GetMounter()
	pluginDir := plugin.host.GetPluginDir(plugin.GetPluginName())
	sourceName, err := mounter.GetDeviceNameFromMount(mountPath, pluginDir)
	if err != nil {
		return nil, err
	}
	glog.V(4).Infof("Found volume %s mounted to %s", sourceName, mountPath)
	cinderVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			Cinder: &v1.CinderVolumeSource{
				VolumeID: sourceName,
			},
		},
	}
	return volume.NewSpecFromVolume(cinderVolume), nil
}

// PersistentVolumePlugin interface.

func (plugin *cinderPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
	}
}

// DeletableVolumePlugin interface.

func (plugin *cinderPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	manager, err := getManager(plugin, spec, nil)
	if err != nil {
		return nil, err
	}
	return newDeleter(spec, plugin, manager)
}

// ProvisionableVolumePlugin interface.

func (plugin *cinderPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	manager, err := getManager(plugin, nil, &options)
	if err != nil {
		return nil, err
	}
	return newProvisioner(options, plugin, manager)
}

// AttachableVolumePlugin interface.

func (plugin *cinderPlugin) NewAttacher() (volume.Attacher, error) {
	return &cinderDiskAttacher{
		cinderVolume: &cinderVolume{
			plugin: plugin,
		},
	}, nil
}

func (plugin *cinderPlugin) NewDetacher() (volume.Detacher, error) {
	return &cinderDiskDetacher{
		cinderVolume: &cinderVolume{
			plugin: plugin,
		},
	}, nil
}

func (plugin *cinderPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter()
	return mount.GetMountRefs(mounter, deviceMountPath)
}
