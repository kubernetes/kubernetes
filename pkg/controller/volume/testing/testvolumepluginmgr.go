/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package testing

import (
	"fmt"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
)

// GetTestVolumePluginMgr creates, initializes, and returns a test volume
// plugin manager.
func GetTestVolumePluginMgr(t *testing.T) (*volume.VolumePluginMgr, *volumetesting.FakeVolumePlugin) {
	plugins := []volume.VolumePlugin{}

	// plugins = append(plugins, aws_ebs.ProbeVolumePlugins()...)
	// plugins = append(plugins, gce_pd.ProbeVolumePlugins()...)
	// plugins = append(plugins, cinder.ProbeVolumePlugins()...)
	volumeTestingPlugins := volumetesting.ProbeVolumePlugins(volume.VolumeConfig{})
	plugins = append(plugins, volumeTestingPlugins...)

	volumePluginMgr := testVolumePluginMgr{}

	if err := volumePluginMgr.InitPlugins(plugins, &volumePluginMgr); err != nil {
		t.Fatalf("Could not initialize volume plugins for Attach/Detach Controller: %+v", err)
	}

	return &volumePluginMgr.VolumePluginMgr, volumeTestingPlugins[0].(*volumetesting.FakeVolumePlugin)
}

type testVolumePluginMgr struct {
	volume.VolumePluginMgr
}

// VolumeHost implementation
// This is an unfortunate requirement of the current factoring of volume plugin
// initializing code. It requires kubelet specific methods used by the mounting
// code to be implemented by all initializers even if the initializer does not
// do mounting (like this attach/detach controller).
// Issue kubernetes/kubernetes/issues/14217 to fix this.
func (vpm *testVolumePluginMgr) GetPluginDir(podUID string) string {
	return ""
}

func (vpm *testVolumePluginMgr) GetPodVolumeDir(podUID types.UID, pluginName, volumeName string) string {
	return ""
}

func (vpm *testVolumePluginMgr) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return ""
}

func (vpm *testVolumePluginMgr) GetKubeClient() internalclientset.Interface {
	return nil
}

func (vpm *testVolumePluginMgr) NewWrapperMounter(volName string, spec volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return nil, fmt.Errorf("NewWrapperMounter not supported by Attach/Detach controller's VolumeHost implementation")
}

func (vpm *testVolumePluginMgr) NewWrapperUnmounter(volName string, spec volume.Spec, podUID types.UID) (volume.Unmounter, error) {
	return nil, fmt.Errorf("NewWrapperUnmounter not supported by Attach/Detach controller's VolumeHost implementation")
}

func (vpm *testVolumePluginMgr) GetCloudProvider() cloudprovider.Interface {
	return &fake.FakeCloud{}
}

func (vpm *testVolumePluginMgr) GetMounter() mount.Interface {
	return nil
}

func (vpm *testVolumePluginMgr) GetWriter() io.Writer {
	return nil
}

func (vpm *testVolumePluginMgr) GetHostName() string {
	return ""
}
