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

package azure_file

import (
	"io/ioutil"
	"os"
	"path"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := ioutil.TempDir(os.TempDir(), "azureFileTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/azure-file")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.Name() != "kubernetes.io/azure-file" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{AzureFile: &api.AzureFileVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &api.PersistentVolume{Spec: api.PersistentVolumeSpec{PersistentVolumeSource: api.PersistentVolumeSource{AzureFile: &api.AzureFileVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
}

func TestGetAccessModes(t *testing.T) {
	tmpDir, err := ioutil.TempDir(os.TempDir(), "azureFileTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/azure-file")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if !contains(plug.GetAccessModes(), api.ReadWriteOnce) || !contains(plug.GetAccessModes(), api.ReadOnlyMany) || !contains(plug.GetAccessModes(), api.ReadWriteMany) {
		t.Errorf("Expected three AccessModeTypes:  %s, %s, and %s", api.ReadWriteOnce, api.ReadOnlyMany, api.ReadWriteMany)
	}
}

func contains(modes []api.PersistentVolumeAccessMode, mode api.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

func TestPlugin(t *testing.T) {
	tmpDir, err := ioutil.TempDir(os.TempDir(), "azurefileTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/azure-file")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &api.Volume{
		Name: "vol1",
		VolumeSource: api.VolumeSource{
			AzureFile: &api.AzureFileVolumeSource{
				SecretName: "secret",
				ShareName:  "share",
			},
		},
	}
	fake := &mount.FakeMounter{}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, err := plug.(*azureFilePlugin).newBuilderInternal(volume.NewSpecFromVolume(spec), pod, &fakeAzureSvc{}, fake)
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}
	volPath := path.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~azure-file/vol1")
	path := builder.GetPath()
	if path != volPath {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err := builder.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	cleaner, err := plug.(*azureFilePlugin).newCleanerInternal("vol1", types.UID("poduid"), &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner")
	}

	if err := cleaner.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "pvA",
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				AzureFile: &api.AzureFileVolumeSource{},
			},
			ClaimRef: &api.ObjectReference{
				Name: "claimA",
			},
		},
	}

	claim := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "claimA",
			Namespace: "nsA",
		},
		Spec: api.PersistentVolumeClaimSpec{
			VolumeName: "pvA",
		},
		Status: api.PersistentVolumeClaimStatus{
			Phase: api.ClaimBound,
		},
	}

	client := fake.NewSimpleClientset(pv, claim)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost("/tmp/fake", client, nil))
	plug, _ := plugMgr.FindPluginByName(azureFilePluginName)

	// readOnly bool is supplied by persistent-claim volume source when its builder creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, _ := plug.NewBuilder(spec, pod, volume.VolumeOptions{})

	if !builder.GetAttributes().ReadOnly {
		t.Errorf("Expected true for builder.IsReadOnly")
	}
}

type fakeAzureSvc struct{}

func (s *fakeAzureSvc) GetAzureCredentials(host volume.VolumeHost, nameSpace, secretName, shareName string) (string, string, error) {
	return "name", "key", nil
}

func TestBuilderAndCleanerTypeAssert(t *testing.T) {
	tmpDir, err := ioutil.TempDir(os.TempDir(), "azurefileTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/azure-file")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &api.Volume{
		Name: "vol1",
		VolumeSource: api.VolumeSource{
			AzureFile: &api.AzureFileVolumeSource{
				SecretName: "secret",
				ShareName:  "share",
			},
		},
	}
	fake := &mount.FakeMounter{}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, err := plug.(*azureFilePlugin).newBuilderInternal(volume.NewSpecFromVolume(spec), pod, &fakeAzureSvc{}, fake)
	if _, ok := builder.(volume.Cleaner); ok {
		t.Errorf("Volume Builder can be type-assert to Cleaner")
	}

	cleaner, err := plug.(*azureFilePlugin).newCleanerInternal("vol1", types.UID("poduid"), &mount.FakeMounter{})
	if _, ok := cleaner.(volume.Builder); ok {
		t.Errorf("Volume Cleaner can be type-assert to Builder")
	}
}
