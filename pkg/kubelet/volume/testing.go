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

package volume

import (
	"os"
	"path"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
)

// FakeHost is useful for testing volume plugins.
type FakeHost struct {
	RootDir      string
	TmpfsRootDir string
	KubeClient   client.Interface
}

func (f *FakeHost) GetPluginDir(pluginName string) string {
	return path.Join(f.RootDir, "plugins", pluginName)
}

func (f *FakeHost) GetPodVolumeDir(podUID types.UID, pluginName, volumeName string) string {
	return path.Join(f.RootDir, "pods", string(podUID), "volumes", pluginName, volumeName)
}

func (f *FakeHost) GetTmpfsPodVolumeDir(podUID types.UID, pluginName, volumeName string) string {
	return path.Join(f.TmpfsRootDir, "pods", string(podUID), "volumes", pluginName, volumeName)
}

func (f *FakeHost) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return path.Join(f.RootDir, "pods", string(podUID), "plugins", pluginName)
}

func (f *FakeHost) GetKubeClient() client.Interface {
	return f.KubeClient
}

// FakePlugin is useful for for testing.  It tries to be a fully compliant
// plugin, but all it does is make empty directories.
// Use as:
//   volume.RegisterPlugin(&FakePlugin{"fake-name"})
type FakePlugin struct {
	PluginName string
	Host       Host
}

var _ Plugin = &FakePlugin{}

func (plugin *FakePlugin) Init(host Host) {
	plugin.Host = host
}

func (plugin *FakePlugin) Name() string {
	return plugin.PluginName
}

func (plugin *FakePlugin) CanSupport(spec *api.Volume) bool {
	// TODO: maybe pattern-match on spec.Name to decide?
	return true
}

func (plugin *FakePlugin) NewBuilder(spec *api.Volume, podUID types.UID) (Builder, error) {
	return &FakeVolume{podUID, spec.Name, plugin}, nil
}

func (plugin *FakePlugin) NewCleaner(volName string, podUID types.UID) (Cleaner, error) {
	return &FakeVolume{podUID, volName, plugin}, nil
}

type FakeVolume struct {
	PodUID  types.UID
	VolName string
	Plugin  *FakePlugin
}

func (fv *FakeVolume) SetUp() error {
	return os.MkdirAll(fv.GetPath(), 0750)
}

func (fv *FakeVolume) GetPath() string {
	return path.Join(fv.Plugin.Host.GetPodVolumeDir(fv.PodUID, EscapePluginName(fv.Plugin.PluginName), fv.VolName))
}

func (fv *FakeVolume) TearDown() error {
	return os.RemoveAll(fv.GetPath())
}
