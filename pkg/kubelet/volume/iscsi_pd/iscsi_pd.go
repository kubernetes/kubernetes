/*
Copyright 2015 Red Hat Inc. All rights reserved.

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

package iscsi_pd

import (
	"fmt"
	"os"
	"path"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume/gce_pd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/golang/glog"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.Plugin {
	return []volume.Plugin{&ISCSIDiskPlugin{nil, false, "", "", ""},
		&ISCSIDiskPlugin{nil, true, "", "", ""}}
}

type ISCSIDiskPlugin struct {
	host       volume.Host
	legacyMode bool // if set, plugin answers to the legacy name
	portal     string
	iqn        string
	lun        string
}

var _ volume.Plugin = &ISCSIDiskPlugin{}

const (
	ISCSIDiskPluginName       = "kubernetes.io/iscsi-pd"
	ISCSIDiskPluginLegacyName = "iscsi-pd"
)

// Abstract interface to PD operations.
type pdManager interface {
	// Attaches the disk to the kubelet's host machine.
	AttachDisk(iscsi *iscsiDisk) error
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(iscsi *iscsiDisk, devicePath string) error
}

type iscsiDisk struct {
	volName    string
	podUID     types.UID
	portal     string
	iqn        string
	readOnly   bool
	lun        string
	fsType     string
	plugin     *ISCSIDiskPlugin
	legacyMode bool
	mounter    mount.Interface
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager pdManager
	exec    exec.Interface
}

func (plugin *ISCSIDiskPlugin) Init(host volume.Host) {
	plugin.host = host
}

func (plugin *ISCSIDiskPlugin) Name() string {
	if plugin.legacyMode {
		return ISCSIDiskPluginLegacyName
	}
	return ISCSIDiskPluginName
}

func (plugin *ISCSIDiskPlugin) CanSupport(spec *api.Volume) bool {
	if plugin.legacyMode {
		// Legacy mode instances can be cleaned up but not created anew.
		return false
	}

	if spec.Source.ISCSIDisk != nil {
		return true
	}
	return false
}

func (plugin *ISCSIDiskPlugin) NewBuilder(spec *api.Volume, podUID types.UID) (volume.Builder, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newBuilderInternal(spec, podUID, &ISCSIDiskUtil{}, mount.New(), exec.New())
}

func (plugin *ISCSIDiskPlugin) newBuilderInternal(spec *api.Volume, podUID types.UID, manager pdManager, mounter mount.Interface, exe exec.Interface) (volume.Builder, error) {
	if plugin.legacyMode {
		// Legacy mode instances can be cleaned up but not created anew.
		return nil, fmt.Errorf("legacy mode: can not create new instances")
	}
	lun := "0"
	if spec.Source.ISCSIDisk.Lun != 0 {
		lun = strconv.Itoa(spec.Source.ISCSIDisk.Lun)
	}
	plugin.portal = spec.Source.ISCSIDisk.Portal
	plugin.iqn = spec.Source.ISCSIDisk.IQN
	plugin.lun = lun
	return &iscsiDisk{
		podUID:     podUID,
		volName:    spec.Name,
		portal:     spec.Source.ISCSIDisk.Portal,
		iqn:        spec.Source.ISCSIDisk.IQN,
		lun:        lun,
		fsType:     spec.Source.ISCSIDisk.FSType,
		readOnly:   spec.Source.ISCSIDisk.ReadOnly,
		exec:       exe,
		manager:    manager,
		mounter:    mounter,
		legacyMode: false,
		plugin:     plugin,
	}, nil
}

func (plugin *ISCSIDiskPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newCleanerInternal(volName, podUID, &ISCSIDiskUtil{}, mount.New(), exec.New())
}

func (plugin *ISCSIDiskPlugin) newCleanerInternal(volName string, podUID types.UID, manager pdManager, mounter mount.Interface, exe exec.Interface) (volume.Cleaner, error) {
	legacy := false
	if plugin.legacyMode {
		legacy = true
	}

	return &iscsiDisk{
		podUID:     podUID,
		volName:    volName,
		legacyMode: legacy,
		manager:    manager,
		mounter:    mounter,
		plugin:     plugin,
		portal:     plugin.portal,
		iqn:        plugin.iqn,
		lun:        plugin.lun,
		exec:       exe,
	}, nil
}

func (iscsi *iscsiDisk) GetPath() string {
	name := ISCSIDiskPluginName
	if iscsi.legacyMode {
		name = ISCSIDiskPluginLegacyName
	}
	// safe to use PodVolumeDir now: volume teardown occurs before pod is cleaned up
	return iscsi.plugin.host.GetPodVolumeDir(iscsi.podUID, volume.EscapePluginName(name), iscsi.volName)
}

func (iscsi *iscsiDisk) SetUp() error {
	if iscsi.legacyMode {
		return fmt.Errorf("legacy mode: can not create new instances")
	}

	globalPDPath := makeGlobalPDName(iscsi.plugin.host, iscsi.portal, iscsi.iqn, iscsi.lun)
	// TODO: handle failed mounts here.
	mountpoint, err := gce_pd.IsMountPoint(iscsi.GetPath())

	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("iscsiPersistentDisk: cannot validate mountpoint")
		return err
	}
	if mountpoint {
		return nil
	}

	if err := iscsi.manager.AttachDisk(iscsi); err != nil {
		glog.Errorf("iSCSIPersistentDisk: failed to attach disk")
		return err
	}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	flags := uintptr(0)
	volPath := iscsi.GetPath()
	if err := os.MkdirAll(volPath, 0750); err != nil {
		return err
	}
	err = iscsi.mounter.Mount(globalPDPath, iscsi.GetPath(), "", mount.FlagBind|flags, "")
	if err != nil {
		glog.Errorf("iSCSIPersistentDisk: failed to bind mount")
		return err
	}

	// make mountpoint rw/ro work as expected
	//FIXME revisit pkg/util/mount and ensure rw/ro is implemented as expected
	if iscsi.readOnly {
		iscsi.execCommand("mount", []string{"-o", "remount,ro", globalPDPath, iscsi.GetPath()})
	} else {
		iscsi.execCommand("mount", []string{"-o", "remount,rw", globalPDPath, iscsi.GetPath()})
	}

	return nil

}

func (iscsi *iscsiDisk) TearDown() error {
	//glog.Infof("iSCSIPersistentDisk: iscsi path %s", iscsi.GetPath())
	mountpoint, err := gce_pd.IsMountPoint(iscsi.GetPath())
	if err != nil {
		glog.Errorf("iSCSIPersistentDisk: cannot validate mountpoint %s", iscsi.GetPath())
		return err
	}
	if !mountpoint {
		return nil
	}

	refs, err := gce_pd.GetMountRefs(iscsi.mounter, iscsi.GetPath())
	if err != nil {
		glog.Errorf("iSCSIPersistentDisk: failed to get reference count %s", iscsi.GetPath())
		return err
	}
	if err := iscsi.mounter.Unmount(iscsi.GetPath(), 0); err != nil {
		glog.Errorf("iSCSIPersistentDisk: failed to umount %s", iscsi.GetPath())
		return err
	}
	// If len(refs) is 1, then all bind mounts have been removed, and the
	// remaining reference is the global mount. It is safe to detach.
	if len(refs) == 1 {
		// pd.pdName is not initially set for volume-cleaners, so set it here.
		devicePath := path.Base(refs[0])
		if err := iscsi.manager.DetachDisk(iscsi, devicePath); err != nil {
			glog.Errorf("iSCSIPersistentDisk: failed to detach disk from %s", devicePath)
			return err
		}
	}
	return nil

}

func (iscsi *iscsiDisk) execCommand(command string, args []string) ([]byte, error) {
	cmd := iscsi.exec.Command(command, args...)
	return cmd.CombinedOutput()
}
