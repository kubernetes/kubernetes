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

// fakeHost is useful for testing volume plugins.
type fakeHost struct {
	rootDir    string
	kubeClient client.Interface
	pluginMgr  PluginMgr
}

func NewFakeHost(rootDir string, kubeClient client.Interface, plugins []Plugin) *fakeHost {
	host := &fakeHost{rootDir: rootDir, kubeClient: kubeClient}
	host.pluginMgr.InitPlugins(plugins, host)
	return host
}

func (f *fakeHost) GetPluginDir(podUID string) string {
	return path.Join(f.rootDir, "plugins", podUID)
}

func (f *fakeHost) GetPodVolumeDir(podUID types.UID, pluginName, volumeName string) string {
	return path.Join(f.rootDir, "pods", string(podUID), "volumes", pluginName, volumeName)
}

func (f *fakeHost) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return path.Join(f.rootDir, "pods", string(podUID), "plugins", pluginName)
}

func (f *fakeHost) GetKubeClient() client.Interface {
	return f.kubeClient
}

func (f *fakeHost) NewWrapperBuilder(spec *api.Volume, podRef *api.ObjectReference) (Builder, error) {
	plug, err := f.pluginMgr.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	return plug.NewBuilder(spec, podRef)
}

func (f *fakeHost) NewWrapperCleaner(spec *api.Volume, podUID types.UID) (Cleaner, error) {
	plug, err := f.pluginMgr.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	return plug.NewCleaner(spec.Name, podUID)
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

func (plugin *FakePlugin) NewBuilder(spec *api.Volume, podRef *api.ObjectReference) (Builder, error) {
	return &FakeVolume{podRef.UID, spec.Name, plugin}, nil
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
	return fv.SetUpAt(fv.GetPath())
}

func (fv *FakeVolume) SetUpAt(dir string) error {
	return os.MkdirAll(dir, 0750)
}

func (fv *FakeVolume) GetPath() string {
	return path.Join(fv.Plugin.Host.GetPodVolumeDir(fv.PodUID, EscapePluginName(fv.Plugin.PluginName), fv.VolName))
}

func (fv *FakeVolume) TearDown() error {
	return fv.TearDownAt(fv.GetPath())
}

func (fv *FakeVolume) TearDownAt(dir string) error {
	return os.RemoveAll(dir)
}
