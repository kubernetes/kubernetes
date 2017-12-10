/*
Copyright 2017 The Kubernetes Authors.

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

package csi

import (
	"fmt"
	"os"
	"path"
	"testing"

	api "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

// create a plugin mgr to load plugins and setup a fake client
func newTestPlugin(t *testing.T) (*csiPlugin, string) {
	tmpDir, err := utiltesting.MkTmpdir("csi-test")
	if err != nil {
		t.Fatalf("can't create temp dir: %v", err)
	}

	fakeClient := fakeclient.NewSimpleClientset()
	host := volumetest.NewFakeVolumeHost(
		tmpDir,
		fakeClient,
		nil,
	)
	plugMgr := &volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, host)

	plug, err := plugMgr.FindPluginByName(csiPluginName)
	if err != nil {
		t.Fatalf("can't find plugin %v", csiPluginName)
	}

	csiPlug, ok := plug.(*csiPlugin)
	if !ok {
		t.Fatalf("cannot assert plugin to be type csiPlugin")
	}

	return csiPlug, tmpDir
}

func makeTestPV(name string, sizeGig int, driverName, volID string) *api.PersistentVolume {
	return &api.PersistentVolume{
		ObjectMeta: meta.ObjectMeta{
			Name:      name,
			Namespace: testns,
		},
		Spec: api.PersistentVolumeSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse(
					fmt.Sprintf("%dGi", sizeGig),
				),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				CSI: &api.CSIPersistentVolumeSource{
					Driver:       driverName,
					VolumeHandle: volID,
					ReadOnly:     false,
				},
			},
		},
	}
}

func TestPluginGetPluginName(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)
	if plug.GetPluginName() != "kubernetes.io/csi" {
		t.Errorf("unexpected plugin name %v", plug.GetPluginName())
	}
}

func TestPluginGetVolumeName(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)
	testCases := []struct {
		name       string
		driverName string
		volName    string
		shouldFail bool
	}{
		{"alphanum names", "testdr", "testvol", false},
		{"mixchar driver", "test.dr.cc", "testvol", false},
		{"mixchar volume", "testdr", "test-vol-name", false},
		{"mixchars all", "test-driver", "test.vol.name", false},
	}

	for _, tc := range testCases {
		t.Logf("testing: %s", tc.name)
		pv := makeTestPV("test-pv", 10, tc.driverName, tc.volName)
		spec := volume.NewSpecFromPersistentVolume(pv, false)
		name, err := plug.GetVolumeName(spec)
		if tc.shouldFail && err == nil {
			t.Fatal("GetVolumeName should fail, but got err=nil")
		}
		if name != fmt.Sprintf("%s%s%s", tc.driverName, volNameSep, tc.volName) {
			t.Errorf("unexpected volume name %s", name)
		}
	}
}

func TestPluginCanSupport(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	pv := makeTestPV("test-pv", 10, testDriver, testVol)
	spec := volume.NewSpecFromPersistentVolume(pv, false)

	if !plug.CanSupport(spec) {
		t.Errorf("should support CSI spec")
	}
}

func TestPluginConstructVolumeSpec(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	testCases := []struct {
		name       string
		specVolID  string
		data       map[string]string
		shouldFail bool
	}{
		{
			name:      "valid spec name",
			specVolID: "test.vol.id",
			data:      map[string]string{volDataKey.specVolID: "test.vol.id", volDataKey.volHandle: "test-vol0", volDataKey.driverName: "test-driver0"},
		},
	}

	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		dir := getTargetPath(testPodUID, tc.specVolID, plug.host)

		// create the data file
		if tc.data != nil {
			mountDir := path.Join(getTargetPath(testPodUID, tc.specVolID, plug.host), "/mount")
			if err := os.MkdirAll(mountDir, 0755); err != nil && !os.IsNotExist(err) {
				t.Errorf("failed to create dir [%s]: %v", mountDir, err)
			}
			if err := saveVolumeData(plug, testPodUID, tc.specVolID, tc.data); err != nil {
				t.Fatal(err)
			}
		}

		// rebuild spec
		spec, err := plug.ConstructVolumeSpec("test-pv", dir)
		if tc.shouldFail {
			if err == nil {
				t.Fatal("expecting ConstructVolumeSpec to fail, but got nil error")
			}
			continue
		}

		volHandle := spec.PersistentVolume.Spec.CSI.VolumeHandle
		if volHandle != tc.data[volDataKey.volHandle] {
			t.Errorf("expected volID %s, got volID %s", tc.data[volDataKey.volHandle], volHandle)
		}

		if spec.Name() != tc.specVolID {
			t.Errorf("Unexpected spec name %s", spec.Name())
		}
	}
}

func TestPluginNewMounter(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	pv := makeTestPV("test-pv", 10, testDriver, testVol)
	mounter, err := plug.NewMounter(
		volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly),
		&api.Pod{ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: testns}},
		volume.VolumeOptions{},
	)
	if err != nil {
		t.Fatalf("Failed to make a new Mounter: %v", err)
	}

	if mounter == nil {
		t.Fatal("failed to create CSI mounter")
	}
	csiMounter := mounter.(*csiMountMgr)

	// validate mounter fields
	if csiMounter.driverName != testDriver {
		t.Error("mounter driver name not set")
	}
	if csiMounter.volumeID != testVol {
		t.Error("mounter volume id not set")
	}
	if csiMounter.pod == nil {
		t.Error("mounter pod not set")
	}
	if csiMounter.podUID == types.UID("") {
		t.Error("mounter podUID mot set")
	}
}

func TestPluginNewUnmounter(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	pv := makeTestPV("test-pv", 10, testDriver, testVol)

	unmounter, err := plug.NewUnmounter(pv.ObjectMeta.Name, testPodUID)
	csiUnmounter := unmounter.(*csiMountMgr)

	if err != nil {
		t.Fatalf("Failed to make a new Mounter: %v", err)
	}

	if csiUnmounter == nil {
		t.Fatal("failed to create CSI mounter")
	}

	if csiUnmounter.podUID != testPodUID {
		t.Error("podUID not set")
	}

}

func TestValidateDriverName(t *testing.T) {
	testCases := []struct {
		name       string
		driverName string
		valid      bool
	}{

		{"ok no punctuations", "comgooglestoragecsigcepd", true},
		{"ok dot only", "io.kubernetes.storage.csi.flex", true},
		{"ok dash only", "io-kubernetes-storage-csi-flex", true},
		{"ok underscore only", "io_kubernetes_storage_csi_flex", true},
		{"ok dot underscores", "io.kubernetes.storage_csi.flex", true},
		{"ok dot dash underscores", "io.kubernetes-storage.csi_flex", true},

		{"invalid length 0", "", false},
		{"invalid length > 63", "comgooglestoragecsigcepdcomgooglestoragecsigcepdcomgooglestoragecsigcepdcomgooglestoragecsigcepd", false},
		{"invalid start char", "_comgooglestoragecsigcepd", false},
		{"invalid end char", "comgooglestoragecsigcepd/", false},
		{"invalid separators", "com/google/storage/csi~gcepd", false},
	}

	for _, tc := range testCases {
		t.Logf("test case: %v", tc.name)
		drValid := isDriverNameValid(tc.driverName)
		if tc.valid != drValid {
			t.Errorf("expecting driverName %s as valid=%t, but got valid=%t", tc.driverName, tc.valid, drValid)
		}
	}
}

func TestPluginNewAttacher(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	attacher, err := plug.NewAttacher()
	if err != nil {
		t.Fatalf("failed to create new attacher: %v", err)
	}

	csiAttacher := attacher.(*csiAttacher)
	if csiAttacher.plugin == nil {
		t.Error("plugin not set for attacher")
	}
	if csiAttacher.k8s == nil {
		t.Error("Kubernetes client not set for attacher")
	}
}

func TestPluginNewDetacher(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	detacher, err := plug.NewDetacher()
	if err != nil {
		t.Fatalf("failed to create new detacher: %v", err)
	}

	csiDetacher := detacher.(*csiAttacher)
	if csiDetacher.plugin == nil {
		t.Error("plugin not set for detacher")
	}
	if csiDetacher.k8s == nil {
		t.Error("Kubernetes client not set for attacher")
	}
}
