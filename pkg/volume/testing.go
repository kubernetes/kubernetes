/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"os"
	"path"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/mount"
)

// fakeVolumeHost is useful for testing volume plugins.
type fakeVolumeHost struct {
	rootDir    string
	kubeClient client.Interface
	pluginMgr  VolumePluginMgr
	cloud      cloudprovider.Interface
}

func NewFakeVolumeHost(rootDir string, kubeClient client.Interface, plugins []VolumePlugin) *fakeVolumeHost {
	host := &fakeVolumeHost{rootDir: rootDir, kubeClient: kubeClient, cloud: nil}
	host.pluginMgr.InitPlugins(plugins, host)
	return host
}

func (f *fakeVolumeHost) GetPluginDir(podUID string) string {
	return path.Join(f.rootDir, "plugins", podUID)
}

func (f *fakeVolumeHost) GetPodVolumeDir(podUID types.UID, pluginName, volumeName string) string {
	return path.Join(f.rootDir, "pods", string(podUID), "volumes", pluginName, volumeName)
}

func (f *fakeVolumeHost) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return path.Join(f.rootDir, "pods", string(podUID), "plugins", pluginName)
}

func (f *fakeVolumeHost) GetKubeClient() client.Interface {
	return f.kubeClient
}

func (f *fakeVolumeHost) GetCloudProvider() cloudprovider.Interface {
	return f.cloud
}

func (f *fakeVolumeHost) NewWrapperBuilder(spec *Spec, pod *api.Pod, opts VolumeOptions, mounter mount.Interface) (Builder, error) {
	plug, err := f.pluginMgr.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	return plug.NewBuilder(spec, pod, opts, mounter)
}

func (f *fakeVolumeHost) NewWrapperCleaner(spec *Spec, podUID types.UID, mounter mount.Interface) (Cleaner, error) {
	plug, err := f.pluginMgr.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	return plug.NewCleaner(spec.Name(), podUID, mounter)
}

func ProbeVolumePlugins(config VolumeConfig) []VolumePlugin {
	if _, ok := config.OtherAttributes["fake-property"]; ok {
		return []VolumePlugin{
			&FakeVolumePlugin{
				PluginName: "fake-plugin",
				Host:       nil,
				// SomeFakeProperty: config.OtherAttributes["fake-property"] -- string, may require parsing by plugin
			},
		}
	}
	return []VolumePlugin{&FakeVolumePlugin{PluginName: "fake-plugin"}}
}

// FakeVolumePlugin is useful for testing.  It tries to be a fully compliant
// plugin, but all it does is make empty directories.
// Use as:
//   volume.RegisterPlugin(&FakePlugin{"fake-name"})
type FakeVolumePlugin struct {
	PluginName string
	Host       VolumeHost
}

var _ VolumePlugin = &FakeVolumePlugin{}
var _ RecyclableVolumePlugin = &FakeVolumePlugin{}

func (plugin *FakeVolumePlugin) Init(host VolumeHost) {
	plugin.Host = host
}

func (plugin *FakeVolumePlugin) Name() string {
	return plugin.PluginName
}

func (plugin *FakeVolumePlugin) CanSupport(spec *Spec) bool {
	// TODO: maybe pattern-match on spec.Name() to decide?
	return true
}

func (plugin *FakeVolumePlugin) NewBuilder(spec *Spec, pod *api.Pod, opts VolumeOptions, mounter mount.Interface) (Builder, error) {
	return &FakeVolume{pod.UID, spec.Name(), plugin}, nil
}

func (plugin *FakeVolumePlugin) NewCleaner(volName string, podUID types.UID, mounter mount.Interface) (Cleaner, error) {
	return &FakeVolume{podUID, volName, plugin}, nil
}

func (plugin *FakeVolumePlugin) NewRecycler(spec *Spec) (Recycler, error) {
	return &fakeRecycler{"/attributesTransferredFromSpec"}, nil
}

func (plugin *FakeVolumePlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{}
}

type FakeVolume struct {
	PodUID  types.UID
	VolName string
	Plugin  *FakeVolumePlugin
}

func (fv *FakeVolume) SetUp() error {
	return fv.SetUpAt(fv.GetPath())
}

func (fv *FakeVolume) SetUpAt(dir string) error {
	return os.MkdirAll(dir, 0750)
}

func (fv *FakeVolume) IsReadOnly() bool {
	return false
}

func (fv *FakeVolume) GetPath() string {
	return path.Join(fv.Plugin.Host.GetPodVolumeDir(fv.PodUID, util.EscapeQualifiedNameForDisk(fv.Plugin.PluginName), fv.VolName))
}

func (fv *FakeVolume) TearDown() error {
	return fv.TearDownAt(fv.GetPath())
}

func (fv *FakeVolume) TearDownAt(dir string) error {
	return os.RemoveAll(dir)
}

type fakeRecycler struct {
	path string
}

func (fr *fakeRecycler) Recycle() error {
	// nil is success, else error
	return nil
}

func (fr *fakeRecycler) GetPath() string {
	return fr.path
}

func NewFakeRecycler(spec *Spec, host VolumeHost, config VolumeConfig) (Recycler, error) {
	if spec.PersistentVolume == nil || spec.PersistentVolume.Spec.HostPath == nil {
		return nil, fmt.Errorf("fakeRecycler only supports spec.PersistentVolume.Spec.HostPath")
	}
	return &fakeRecycler{
		path: spec.PersistentVolume.Spec.HostPath.Path,
	}, nil
}
