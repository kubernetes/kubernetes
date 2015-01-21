/*
Copyright 2014 Google Inc. All rights reserved.

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

package empty_dir

import (
	"fmt"
	"os"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.Plugin {
	return []volume.Plugin{&emptyDirPlugin{nil, false}, &emptyDirPlugin{nil, true}}
}

type emptyDirPlugin struct {
	host       volume.Host
	legacyMode bool // if set, plugin answers to the legacy name
}

var _ volume.Plugin = &emptyDirPlugin{}

const (
	emptyDirPluginName       = "kubernetes.io/empty-dir"
	emptyDirPluginLegacyName = "empty"
)

func (plugin *emptyDirPlugin) Init(host volume.Host) {
	plugin.host = host
}

func (plugin *emptyDirPlugin) Name() string {
	if plugin.legacyMode {
		return emptyDirPluginLegacyName
	}
	return emptyDirPluginName
}

func (plugin *emptyDirPlugin) CanSupport(spec *api.Volume) bool {
	if plugin.legacyMode {
		// Legacy mode instances can be cleaned up but not created anew.
		return false
	}

	if util.AllPtrFieldsNil(&spec.Source) {
		return true
	}
	if spec.Source.EmptyDir != nil {
		return true
	}
	return false
}

func (plugin *emptyDirPlugin) NewBuilder(spec *api.Volume, podUID types.UID) (volume.Builder, error) {
	if plugin.legacyMode {
		// Legacy mode instances can be cleaned up but not created anew.
		return nil, fmt.Errorf("legacy mode: can not create new instances")
	}
	return &emptyDir{podUID, spec.Name, plugin, false}, nil
}

func (plugin *emptyDirPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	legacy := false
	if plugin.legacyMode {
		legacy = true
	}
	return &emptyDir{podUID, volName, plugin, legacy}, nil
}

// EmptyDir volumes are temporary directories exposed to the pod.
// These do not persist beyond the lifetime of a pod.
type emptyDir struct {
	podUID     types.UID
	volName    string
	plugin     *emptyDirPlugin
	legacyMode bool
}

// SetUp creates new directory.
func (ed *emptyDir) SetUp() error {
	if ed.legacyMode {
		return fmt.Errorf("legacy mode: can not create new instances")
	}
	path := ed.GetPath()
	return os.MkdirAll(path, 0750)
}

func (ed *emptyDir) GetPath() string {
	name := emptyDirPluginName
	if ed.legacyMode {
		name = emptyDirPluginLegacyName
	}
	return ed.plugin.host.GetPodVolumeDir(ed.podUID, volume.EscapePluginName(name), ed.volName)
}

// TearDown simply deletes everything in the directory.
func (ed *emptyDir) TearDown() error {
	tmpDir, err := volume.RenameDirectory(ed.GetPath(), ed.volName+".deleting~")
	if err != nil {
		return err
	}
	err = os.RemoveAll(tmpDir)
	if err != nil {
		return err
	}
	return nil
}
