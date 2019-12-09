// +build !providerless

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
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/Azure/go-autorest/autorest/to"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	fakecloud "k8s.io/cloud-provider/fake"
	"k8s.io/utils/mount"

	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/legacy-cloud-providers/azure"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := ioutil.TempDir(os.TempDir(), "azureFileTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

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
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{AzureFile: &v1.AzureFilePersistentVolumeSource{}}}}}) {
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
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/azure-file")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if !volumetest.ContainsAccessMode(plug.GetAccessModes(), v1.ReadWriteOnce) || !volumetest.ContainsAccessMode(plug.GetAccessModes(), v1.ReadOnlyMany) || !volumetest.ContainsAccessMode(plug.GetAccessModes(), v1.ReadWriteMany) {
		t.Errorf("Expected three AccessModeTypes:  %s, %s, and %s", v1.ReadWriteOnce, v1.ReadOnlyMany, v1.ReadWriteMany)
	}
}

func getAzureTestCloud(t *testing.T) *azure.Cloud {
	config := `{
                "aadClientId": "--aad-client-id--",
                "aadClientSecret": "--aad-client-secret--"
        }`
	configReader := strings.NewReader(config)
	azureCloud, err := azure.NewCloudWithoutFeatureGates(configReader)
	if err != nil {
		t.Error(err)
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
	testPlugin(t, tmpDir, volumetest.NewFakeVolumeHostWithCloudProvider(t, tmpDir, nil, nil, getAzureTestCloud(t)))
}

func TestPluginWithoutCloudProvider(t *testing.T) {
	tmpDir := getTestTempDir(t)
	defer os.RemoveAll(tmpDir)
	testPlugin(t, tmpDir, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))
}

func TestPluginWithOtherCloudProvider(t *testing.T) {
	tmpDir := getTestTempDir(t)
	defer os.RemoveAll(tmpDir)
	cloud := &fakecloud.Cloud{}
	testPlugin(t, tmpDir, volumetest.NewFakeVolumeHostWithCloudProvider(t, tmpDir, nil, nil, cloud))
}

func testPlugin(t *testing.T, tmpDir string, volumeHost volume.VolumeHost) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumeHost)

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
	fake := mount.NewFakeMounter(nil)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.(*azureFilePlugin).newMounterInternal(volume.NewSpecFromVolume(spec), pod, &fakeAzureSvc{}, fake)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}
	volPath := filepath.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~azure-file/vol1")
	path := mounter.GetPath()
	if path != volPath {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err := mounter.SetUp(volume.MounterArgs{}); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	unmounter, err := plug.(*azureFilePlugin).newUnmounterInternal("vol1", types.UID("poduid"), mount.NewFakeMounter(nil))
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
		t.Errorf("TearDown() failed: %v", err)
	}
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pvA",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AzureFile: &v1.AzureFilePersistentVolumeSource{},
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
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, "/tmp/fake", client, nil))
	plug, _ := plugMgr.FindPluginByName(azureFilePluginName)

	// readOnly bool is supplied by persistent-claim volume source when its mounter creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, _ := plug.NewMounter(spec, pod, volume.VolumeOptions{})
	if mounter == nil {
		t.Fatalf("Got a nil Mounter")
	}

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
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

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
	fake := mount.NewFakeMounter(nil)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.(*azureFilePlugin).newMounterInternal(volume.NewSpecFromVolume(spec), pod, &fakeAzureSvc{}, fake)
	if err != nil {
		t.Errorf("MounterInternal() failed: %v", err)
	}
	if _, ok := mounter.(volume.Unmounter); ok {
		t.Errorf("Volume Mounter can be type-assert to Unmounter")
	}

	unmounter, err := plug.(*azureFilePlugin).newUnmounterInternal("vol1", types.UID("poduid"), mount.NewFakeMounter(nil))
	if err != nil {
		t.Errorf("MounterInternal() failed: %v", err)
	}
	if _, ok := unmounter.(volume.Mounter); ok {
		t.Errorf("Volume Unmounter can be type-assert to Mounter")
	}
}

type testcase struct {
	name      string
	defaultNs string
	spec      *volume.Spec
	// Expected return of the test
	expectedName  string
	expectedNs    string
	expectedError error
}

func TestGetSecretNameAndNamespaceForPV(t *testing.T) {
	secretNs := "ns"
	tests := []testcase{
		{
			name:      "persistent volume source",
			defaultNs: "default",
			spec: &volume.Spec{
				PersistentVolume: &v1.PersistentVolume{
					Spec: v1.PersistentVolumeSpec{
						PersistentVolumeSource: v1.PersistentVolumeSource{
							AzureFile: &v1.AzureFilePersistentVolumeSource{
								ShareName:       "share",
								SecretName:      "name",
								SecretNamespace: &secretNs,
							},
						},
					},
				},
			},
			expectedName:  "name",
			expectedNs:    "ns",
			expectedError: nil,
		},
		{
			name:      "persistent volume source without namespace",
			defaultNs: "default",
			spec: &volume.Spec{
				PersistentVolume: &v1.PersistentVolume{
					Spec: v1.PersistentVolumeSpec{
						PersistentVolumeSource: v1.PersistentVolumeSource{
							AzureFile: &v1.AzureFilePersistentVolumeSource{
								ShareName:  "share",
								SecretName: "name",
							},
						},
					},
				},
			},
			expectedName:  "name",
			expectedNs:    "default",
			expectedError: nil,
		},
		{
			name:      "pod volume source",
			defaultNs: "default",
			spec: &volume.Spec{
				Volume: &v1.Volume{
					VolumeSource: v1.VolumeSource{
						AzureFile: &v1.AzureFileVolumeSource{
							ShareName:  "share",
							SecretName: "name",
						},
					},
				},
			},
			expectedName:  "name",
			expectedNs:    "default",
			expectedError: nil,
		},
	}
	for _, testcase := range tests {
		resultName, resultNs, err := getSecretNameAndNamespace(testcase.spec, testcase.defaultNs)
		if err != testcase.expectedError || resultName != testcase.expectedName || resultNs != testcase.expectedNs {
			t.Errorf("%s failed: expected err=%v ns=%q name=%q, got %v/%q/%q", testcase.name, testcase.expectedError, testcase.expectedNs, testcase.expectedName,
				err, resultNs, resultName)
		}
	}

}

func TestAppendDefaultMountOptions(t *testing.T) {
	tests := []struct {
		options  []string
		fsGroup  *int64
		expected []string
	}{
		{
			options: []string{"dir_mode=0777"},
			fsGroup: nil,
			expected: []string{"dir_mode=0777",
				fmt.Sprintf("%s=%s", fileMode, defaultFileMode),
				fmt.Sprintf("%s=%s", vers, defaultVers)},
		},
		{
			options: []string{"file_mode=0777"},
			fsGroup: to.Int64Ptr(0),
			expected: []string{"file_mode=0777",
				fmt.Sprintf("%s=%s", dirMode, defaultDirMode),
				fmt.Sprintf("%s=%s", vers, defaultVers),
				fmt.Sprintf("%s=0", gid)},
		},
		{
			options: []string{"vers=2.1"},
			fsGroup: to.Int64Ptr(1000),
			expected: []string{"vers=2.1",
				fmt.Sprintf("%s=%s", fileMode, defaultFileMode),
				fmt.Sprintf("%s=%s", dirMode, defaultDirMode),
				fmt.Sprintf("%s=1000", gid)},
		},
		{
			options: []string{""},
			expected: []string{"", fmt.Sprintf("%s=%s",
				fileMode, defaultFileMode),
				fmt.Sprintf("%s=%s", dirMode, defaultDirMode),
				fmt.Sprintf("%s=%s", vers, defaultVers)},
		},
		{
			options:  []string{"file_mode=0777", "dir_mode=0777"},
			expected: []string{"file_mode=0777", "dir_mode=0777", fmt.Sprintf("%s=%s", vers, defaultVers)},
		},
		{
			options: []string{"gid=2000"},
			fsGroup: to.Int64Ptr(1000),
			expected: []string{"gid=2000",
				fmt.Sprintf("%s=%s", fileMode, defaultFileMode),
				fmt.Sprintf("%s=%s", dirMode, defaultDirMode),
				"vers=3.0"},
		},
	}

	for _, test := range tests {
		result := appendDefaultMountOptions(test.options, test.fsGroup)
		if !reflect.DeepEqual(result, test.expected) {
			t.Errorf("input: %q, appendDefaultMountOptions result: %q, expected: %q", test.options, result, test.expected)
		}
	}
}
