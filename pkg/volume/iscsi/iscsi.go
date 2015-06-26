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

package iscsi

import (
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
	return []volume.VolumePlugin{&iscsiPlugin{nil, exec.New()}}
}

type iscsiPlugin struct {
	host volume.VolumeHost
	exe  exec.Interface
}

var _ volume.VolumePlugin = &iscsiPlugin{}
var _ volume.PersistentVolumePlugin = &iscsiPlugin{}

const (
	iscsiPluginName = "kubernetes.io/iscsi"
)

func (plugin *iscsiPlugin) Init(host volume.VolumeHost) {
	plugin.host = host
}

func (plugin *iscsiPlugin) Name() string {
	return iscsiPluginName
}

func (plugin *iscsiPlugin) CanSupport(spec *volume.Spec) bool {
	if (spec.Volume != nil && spec.Volume.ISCSI == nil) || (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.ISCSI == nil) {
		return false
	}
	// TODO:  turn this into a func so CanSupport can be unit tested without
	// having to make system calls
	// see if iscsiadm is there
	_, err := plugin.execCommand("iscsiadm", []string{"-h"})
	if err == nil {
		return true
	}

	return false
}

func (plugin *iscsiPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
	}
}

func (plugin *iscsiPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions, mounter mount.Interface) (volume.Builder, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newBuilderInternal(spec, pod.UID, &ISCSIUtil{}, mounter)
}

func (plugin *iscsiPlugin) newBuilderInternal(spec *volume.Spec, podUID types.UID, manager diskManager, mounter mount.Interface) (volume.Builder, error) {
	// iscsi volumes used directly in a pod have a ReadOnly flag set by the pod author.
	// iscsi volumes used as a PersistentVolume gets the ReadOnly flag indirectly through the persistent-claim volume used to mount the PV
	var readOnly bool
	var iscsi *api.ISCSIVolumeSource
	if spec.Volume != nil && spec.Volume.ISCSI != nil {
		iscsi = spec.Volume.ISCSI
		readOnly = iscsi.ReadOnly
	} else {
		iscsi = spec.PersistentVolume.Spec.ISCSI
		readOnly = spec.ReadOnly
	}

	lun := strconv.Itoa(iscsi.Lun)

	return &iscsiDiskBuilder{
		iscsiDisk: &iscsiDisk{
			podUID:  podUID,
			volName: spec.Name(),
			portal:  iscsi.TargetPortal,
			iqn:     iscsi.IQN,
			lun:     lun,
			manager: manager,
			mounter: mounter,
			plugin:  plugin},
		fsType:   iscsi.FSType,
		readOnly: readOnly,
	}, nil
}

func (plugin *iscsiPlugin) NewCleaner(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newCleanerInternal(volName, podUID, &ISCSIUtil{}, mounter)
}

func (plugin *iscsiPlugin) newCleanerInternal(volName string, podUID types.UID, manager diskManager, mounter mount.Interface) (volume.Cleaner, error) {
	return &iscsiDiskCleaner{&iscsiDisk{
		podUID:  podUID,
		volName: volName,
		manager: manager,
		mounter: mounter,
		plugin:  plugin,
	}}, nil
}

func (plugin *iscsiPlugin) execCommand(command string, args []string) ([]byte, error) {
	cmd := plugin.exe.Command(command, args...)
	return cmd.CombinedOutput()
}

type iscsiDisk struct {
	volName string
	podUID  types.UID
	portal  string
	iqn     string
	lun     string
	plugin  *iscsiPlugin
	mounter mount.Interface
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager diskManager
}

func (iscsi *iscsiDisk) GetPath() string {
	name := iscsiPluginName
	// safe to use PodVolumeDir now: volume teardown occurs before pod is cleaned up
	return iscsi.plugin.host.GetPodVolumeDir(iscsi.podUID, util.EscapeQualifiedNameForDisk(name), iscsi.volName)
}

type iscsiDiskBuilder struct {
	*iscsiDisk
	readOnly bool
	fsType   string
}

var _ volume.Builder = &iscsiDiskBuilder{}

func (b *iscsiDiskBuilder) SetUp() error {
	return b.SetUpAt(b.GetPath())
}

func (b *iscsiDiskBuilder) SetUpAt(dir string) error {
	// diskSetUp checks mountpoints and prevent repeated calls
	err := diskSetUp(b.manager, *b, dir, b.mounter)
	if err != nil {
		glog.Errorf("iscsi: failed to setup")
	}
	return err
}

type iscsiDiskCleaner struct {
	*iscsiDisk
}

var _ volume.Cleaner = &iscsiDiskCleaner{}

func (b *iscsiDiskBuilder) IsReadOnly() bool {
	return b.readOnly
}

// Unmounts the bind mount, and detaches the disk only if the disk
// resource was the last reference to that disk on the kubelet.
func (c *iscsiDiskCleaner) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *iscsiDiskCleaner) TearDownAt(dir string) error {
	return diskTearDown(c.manager, *c, dir, c.mounter)
}
