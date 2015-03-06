/*
Copyright 2015 Google Inc. All rights reserved.

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
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/pd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
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

	if spec.ISCSIDisk != nil {
		return true
	}
	return false
}

func (plugin *ISCSIDiskPlugin) NewBuilder(spec *api.Volume, podUID types.UID) (volume.Builder, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newBuilderInternal(spec, podUID, &ISCSIDiskUtil{}, mount.New(), exec.New())
}

func (plugin *ISCSIDiskPlugin) newBuilderInternal(spec *api.Volume, podUID types.UID, manager disk.PDManager, mounter mount.Interface, exe exec.Interface) (volume.Builder, error) {
	if plugin.legacyMode {
		// Legacy mode instances can be cleaned up but not created anew.
		return nil, fmt.Errorf("legacy mode: can not create new instances")
	}
	lun := "0"
	if spec.ISCSIDisk.Lun != 0 {
		lun = strconv.Itoa(spec.ISCSIDisk.Lun)
	}
	plugin.portal = spec.ISCSIDisk.Portal
	plugin.iqn = spec.ISCSIDisk.IQN
	plugin.lun = lun
	return &iscsiDisk{
		podUID:     podUID,
		volName:    spec.Name,
		portal:     spec.ISCSIDisk.Portal,
		iqn:        spec.ISCSIDisk.IQN,
		lun:        lun,
		fsType:     spec.ISCSIDisk.FSType,
		readOnly:   spec.ISCSIDisk.ReadOnly,
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

func (plugin *ISCSIDiskPlugin) newCleanerInternal(volName string, podUID types.UID, manager disk.PDManager, mounter mount.Interface, exe exec.Interface) (volume.Cleaner, error) {
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
	manager disk.PDManager
	exec    exec.Interface
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

	err := disk.CommonPDSetUp(iscsi.manager, *iscsi, iscsi.GetPath(), iscsi.mounter)
	if err != nil {
		glog.Errorf("iSCSIPersistentDisk: failed to setup")
		return err
	}
	globalPDPath := iscsi.manager.MakeGlobalPDName(*iscsi)
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
	return disk.CommonPDTearDown(iscsi.manager, *iscsi, iscsi.GetPath(), iscsi.mounter)
}

func (iscsi *iscsiDisk) execCommand(command string, args []string) ([]byte, error) {
	cmd := iscsi.exec.Command(command, args...)
	return cmd.CombinedOutput()
}
