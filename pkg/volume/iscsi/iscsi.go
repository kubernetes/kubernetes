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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/golang/glog"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&ISCSIPlugin{nil, exec.New()}}
}

type ISCSIPlugin struct {
	host volume.VolumeHost
	exe  exec.Interface
}

var _ volume.VolumePlugin = &ISCSIPlugin{}

const (
	ISCSIPluginName = "kubernetes.io/iscsi"
)

func (plugin *ISCSIPlugin) Init(host volume.VolumeHost) {
	plugin.host = host
}

func (plugin *ISCSIPlugin) Name() string {
	return ISCSIPluginName
}

func (plugin *ISCSIPlugin) CanSupport(spec *volume.Spec) bool {
	if spec.VolumeSource.ISCSI == nil {
		return false
	}
	// see if iscsiadm is there
	_, err := plugin.execCommand("iscsiadm", []string{"-h"})
	if err == nil {
		return true
	}

	return false
}

func (plugin *ISCSIPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
	}
}

func (plugin *ISCSIPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions, mounter mount.Interface) (volume.Builder, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newBuilderInternal(spec, pod.UID, &ISCSIUtil{}, mounter)
}

func (plugin *ISCSIPlugin) newBuilderInternal(spec *volume.Spec, podUID types.UID, manager diskManager, mounter mount.Interface) (volume.Builder, error) {
	iscsi := spec.VolumeSource.ISCSI
	lun := strconv.Itoa(iscsi.Lun)

	return &iscsiDisk{
		podUID:   podUID,
		volName:  spec.Name,
		portal:   iscsi.TargetPortal,
		iqn:      iscsi.IQN,
		lun:      lun,
		fsType:   iscsi.FSType,
		readOnly: iscsi.ReadOnly,
		manager:  manager,
		mounter:  mounter,
		plugin:   plugin,
	}, nil
}

func (plugin *ISCSIPlugin) NewCleaner(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newCleanerInternal(volName, podUID, &ISCSIUtil{}, mounter)
}

func (plugin *ISCSIPlugin) newCleanerInternal(volName string, podUID types.UID, manager diskManager, mounter mount.Interface) (volume.Cleaner, error) {
	return &iscsiDisk{
		podUID:  podUID,
		volName: volName,
		manager: manager,
		mounter: mounter,
		plugin:  plugin,
	}, nil
}

func (plugin *ISCSIPlugin) execCommand(command string, args []string) ([]byte, error) {
	cmd := plugin.exe.Command(command, args...)
	return cmd.CombinedOutput()
}

type iscsiDisk struct {
	volName  string
	podUID   types.UID
	portal   string
	iqn      string
	readOnly bool
	lun      string
	fsType   string
	plugin   *ISCSIPlugin
	mounter  mount.Interface
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager diskManager
}

func (iscsi *iscsiDisk) GetPath() string {
	name := ISCSIPluginName
	// safe to use PodVolumeDir now: volume teardown occurs before pod is cleaned up
	return iscsi.plugin.host.GetPodVolumeDir(iscsi.podUID, util.EscapeQualifiedNameForDisk(name), iscsi.volName)
}

func (iscsi *iscsiDisk) SetUp() error {
	return iscsi.SetUpAt(iscsi.GetPath())
}

func (iscsi *iscsiDisk) SetUpAt(dir string) error {
	// diskSetUp checks mountpoints and prevent repeated calls
	err := diskSetUp(iscsi.manager, *iscsi, dir, iscsi.mounter)
	if err != nil {
		glog.Errorf("iscsi: failed to setup")
		return err
	}
	globalPDPath := iscsi.manager.MakeGlobalPDName(*iscsi)
	var options []string
	if iscsi.readOnly {
		options = []string{"remount", "ro"}
	} else {
		options = []string{"remount", "rw"}
	}
	return iscsi.mounter.Mount(globalPDPath, dir, "", options)
}

// Unmounts the bind mount, and detaches the disk only if the disk
// resource was the last reference to that disk on the kubelet.
func (iscsi *iscsiDisk) TearDown() error {
	return iscsi.TearDownAt(iscsi.GetPath())
}

func (iscsi *iscsiDisk) TearDownAt(dir string) error {
	return diskTearDown(iscsi.manager, *iscsi, dir, iscsi.mounter)
}
