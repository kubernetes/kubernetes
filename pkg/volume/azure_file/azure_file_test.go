/*
Copyright 2016 The Kubernetes Authors.

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
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
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
	if plug.GetPluginName() != "kubernetes.io/azure-file" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{AzureFile: &v1.AzureFileVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{AzureFile: &v1.AzureFileVolumeSource{}}}}}) {
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
	if !contains(plug.GetAccessModes(), v1.ReadWriteOnce) || !contains(plug.GetAccessModes(), v1.ReadOnlyMany) || !contains(plug.GetAccessModes(), v1.ReadWriteMany) {
		t.Errorf("Expected three AccessModeTypes:  %s, %s, and %s", v1.ReadWriteOnce, v1.ReadOnlyMany, v1.ReadWriteMany)
	}
}

func contains(modes []v1.PersistentVolumeAccessMode, mode v1.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

func getAzureTestCloud(t *testing.T) *azure.Cloud {
	config := `{
                "aadClientId": "--aad-client-id--",
                "aadClientSecret": "--aad-client-secret--"
        }`
	configReader := strings.NewReader(config)
	cloud, err := azure.NewCloud(configReader)
	if err != nil {
		t.Error(err)
	}
	azureCloud, ok := cloud.(*azure.Cloud)
	if !ok {
		t.Error("NewCloud returned incorrect type")
	}
	return azureCloud
}

func getTestTempDir(t *testing.T) string {
	tmpDir, err := ioutil.TempDir(os.TempDir(), "azurefileTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	return tmpDir
}

func TestPluginAzureCloudProvider(t *testing.T) {
	tmpDir := getTestTempDir(t)
	defer os.RemoveAll(tmpDir)
	testPlugin(t, tmpDir, volumetest.NewFakeVolumeHostWithCloudProvider(tmpDir, nil, nil, getAzureTestCloud(t)))
}

func TestPluginWithoutCloudProvider(t *testing.T) {
	tmpDir := getTestTempDir(t)
	defer os.RemoveAll(tmpDir)
	testPlugin(t, tmpDir, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))
}

func TestPluginWithOtherCloudProvider(t *testing.T) {
	tmpDir := getTestTempDir(t)
	defer os.RemoveAll(tmpDir)
	cloud := &fakecloud.FakeCloud{}
	testPlugin(t, tmpDir, volumetest.NewFakeVolumeHostWithCloudProvider(tmpDir, nil, nil, cloud))
}

func testPlugin(t *testing.T, tmpDir string, volumeHost volume.VolumeHost) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumeHost)

	plug, err := plugMgr.FindPluginByName("kubernetes.io/azure-file")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			AzureFile: &v1.AzureFileVolumeSource{
				SecretName: "secret",
				ShareName:  "share",
			},
		},
	}
	fake := &mount.FakeMounter{}
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.(*azureFilePlugin).newMounterInternal(volume.NewSpecFromVolume(spec), pod, &fakeAzureSvc{}, fake)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}
	volPath := path.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~azure-file/vol1")
	path := mounter.GetPath()
	if path != volPath {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err := mounter.SetUp(nil); err != nil {
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

	unmounter, err := plug.(*azureFilePlugin).newUnmounterInternal("vol1", types.UID("poduid"), &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter")
	}

	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pvA",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AzureFile: &v1.AzureFileVolumeSource{},
			},
			ClaimRef: &v1.ObjectReference{
				Name: "claimA",
			},
		},
	}

	claim := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claimA",
			Namespace: "nsA",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			VolumeName: "pvA",
		},
		Status: v1.PersistentVolumeClaimStatus{
			Phase: v1.ClaimBound,
		},
	}

	client := fake.NewSimpleClientset(pv, claim)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost("/tmp/fake", client, nil))
	plug, _ := plugMgr.FindPluginByName(azureFilePluginName)

	// readOnly bool is supplied by persistent-claim volume source when its mounter creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, _ := plug.NewMounter(spec, pod, volume.VolumeOptions{})

	if !mounter.GetAttributes().ReadOnly {
		t.Errorf("Expected true for mounter.IsReadOnly")
	}
}

type fakeAzureSvc struct{}

func (s *fakeAzureSvc) GetAzureCredentials(host volume.VolumeHost, nameSpace, secretName string) (string, string, error) {
	return "name", "key", nil
}
func (s *fakeAzureSvc) SetAzureCredentials(host volume.VolumeHost, nameSpace, accountName, accountKey string) (string, error) {
	return "secret", nil
}

func TestMounterAndUnmounterTypeAssert(t *testing.T) {
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
	spec := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			AzureFile: &v1.AzureFileVolumeSource{
				SecretName: "secret",
				ShareName:  "share",
			},
		},
	}
	fake := &mount.FakeMounter{}
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.(*azureFilePlugin).newMounterInternal(volume.NewSpecFromVolume(spec), pod, &fakeAzureSvc{}, fake)
	if _, ok := mounter.(volume.Unmounter); ok {
		t.Errorf("Volume Mounter can be type-assert to Unmounter")
	}

	unmounter, err := plug.(*azureFilePlugin).newUnmounterInternal("vol1", types.UID("poduid"), &mount.FakeMounter{})
	if _, ok := unmounter.(volume.Mounter); ok {
		t.Errorf("Volume Unmounter can be type-assert to Mounter")
	}
}
