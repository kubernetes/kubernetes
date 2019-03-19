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
	"path/filepath"
	"testing"

	api "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	fakecsi "k8s.io/csi-api/pkg/client/clientset/versioned/fake"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

// create a plugin mgr to load plugins and setup a fake client
func newTestPlugin(t *testing.T, client *fakeclient.Clientset, csiClient *fakecsi.Clientset) (*csiPlugin, string) {
	tmpDir, err := utiltesting.MkTmpdir("csi-test")
	if err != nil {
		t.Fatalf("can't create temp dir: %v", err)
	}

	if client == nil {
		client = fakeclient.NewSimpleClientset()
	}
	if csiClient == nil {
		csiClient = fakecsi.NewSimpleClientset()
	}
	host := volumetest.NewFakeVolumeHostWithCSINodeName(
		tmpDir,
		client,
		csiClient,
		nil,
		"fakeNode",
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

	if utilfeature.DefaultFeatureGate.Enabled(features.CSIDriverRegistry) {
		// Wait until the informer in CSI volume plugin has all CSIDrivers.
		wait.PollImmediate(testInformerSyncPeriod, testInformerSyncTimeout, func() (bool, error) {
			return csiPlug.csiDriverInformer.Informer().HasSynced(), nil
		})
	}

	return csiPlug, tmpDir
}

func makeTestPV(name string, sizeGig int, driverName, volID string) *api.PersistentVolume {
	return &api.PersistentVolume{
		ObjectMeta: meta.ObjectMeta{
			Name: name,
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

func registerFakePlugin(pluginName, endpoint string, versions []string, t *testing.T) {
	csiDrivers = csiDriversStore{driversMap: map[string]csiDriver{}}
	highestSupportedVersions, err := highestSupportedVersion(versions)
	if err != nil {
		t.Fatalf("unexpected error parsing versions (%v) for pluginName % q endpoint %q: %#v", versions, pluginName, endpoint, err)
	}

	csiDrivers.driversMap[pluginName] = csiDriver{driverName: pluginName, driverEndpoint: endpoint, highestSupportedVersion: highestSupportedVersions}
}

func TestPluginGetPluginName(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)
	if plug.GetPluginName() != "kubernetes.io/csi" {
		t.Errorf("unexpected plugin name %v", plug.GetPluginName())
	}
}

func TestPluginGetVolumeName(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
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
		registerFakePlugin(tc.driverName, "endpoint", []string{"0.3.0"}, t)
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
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	registerFakePlugin(testDriver, "endpoint", []string{"1.0.0"}, t)
	pv := makeTestPV("test-pv", 10, testDriver, testVol)
	spec := volume.NewSpecFromPersistentVolume(pv, false)

	if !plug.CanSupport(spec) {
		t.Errorf("should support CSI spec")
	}
}

func TestPluginConstructVolumeSpec(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
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
			if err := saveVolumeData(path.Dir(mountDir), volDataFileName, tc.data); err != nil {
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

		if spec.PersistentVolume.Spec.VolumeMode == nil {
			t.Fatalf("Volume mode has not been set.")
		}

		if *spec.PersistentVolume.Spec.VolumeMode != api.PersistentVolumeFilesystem {
			t.Errorf("Unexpected volume mode %q", *spec.PersistentVolume.Spec.VolumeMode)
		}

		if spec.Name() != tc.specVolID {
			t.Errorf("Unexpected spec name %s", spec.Name())
		}
	}
}

func TestPluginNewMounter(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	registerFakePlugin(testDriver, "endpoint", []string{"1.2.0"}, t)
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
	if string(csiMounter.driverName) != testDriver {
		t.Error("mounter driver name not set")
	}
	if csiMounter.volumeID != testVol {
		t.Error("mounter volume id not set")
	}
	if csiMounter.pod == nil {
		t.Error("mounter pod not set")
	}
	if csiMounter.podUID == types.UID("") {
		t.Error("mounter podUID not set")
	}
	csiClient, err := csiMounter.csiClientGetter.Get()
	if csiClient == nil {
		t.Error("mounter csiClient is nil")
	}

	// ensure data file is created
	dataDir := path.Dir(mounter.GetPath())
	dataFile := filepath.Join(dataDir, volDataFileName)
	if _, err := os.Stat(dataFile); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("data file not created %s", dataFile)
		} else {
			t.Fatal(err)
		}
	}
}

func TestPluginNewUnmounter(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	registerFakePlugin(testDriver, "endpoint", []string{"1.0.0"}, t)
	pv := makeTestPV("test-pv", 10, testDriver, testVol)

	// save the data file to re-create client
	dir := path.Join(getTargetPath(testPodUID, pv.ObjectMeta.Name, plug.host), "/mount")
	if err := os.MkdirAll(dir, 0755); err != nil && !os.IsNotExist(err) {
		t.Errorf("failed to create dir [%s]: %v", dir, err)
	}

	if err := saveVolumeData(
		path.Dir(dir),
		volDataFileName,
		map[string]string{
			volDataKey.specVolID:  pv.ObjectMeta.Name,
			volDataKey.driverName: testDriver,
			volDataKey.volHandle:  testVol,
		},
	); err != nil {
		t.Fatalf("failed to save volume data: %v", err)
	}

	// test unmounter
	unmounter, err := plug.NewUnmounter(pv.ObjectMeta.Name, testPodUID)
	csiUnmounter := unmounter.(*csiMountMgr)

	if err != nil {
		t.Fatalf("Failed to make a new Unmounter: %v", err)
	}

	if csiUnmounter == nil {
		t.Fatal("failed to create CSI Unmounter")
	}

	if csiUnmounter.podUID != testPodUID {
		t.Error("podUID not set")
	}

	csiClient, err := csiUnmounter.csiClientGetter.Get()
	if csiClient == nil {
		t.Error("mounter csiClient is nil")
	}
}

func TestPluginNewAttacher(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
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
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
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
		t.Error("Kubernetes client not set for detacher")
	}
}

func TestPluginNewBlockMapper(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	registerFakePlugin(testDriver, "endpoint", []string{"1.0.0"}, t)
	pv := makeTestPV("test-block-pv", 10, testDriver, testVol)
	mounter, err := plug.NewBlockVolumeMapper(
		volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly),
		&api.Pod{ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: testns}},
		volume.VolumeOptions{},
	)
	if err != nil {
		t.Fatalf("Failed to make a new BlockMapper: %v", err)
	}

	if mounter == nil {
		t.Fatal("failed to create CSI BlockMapper, mapper is nill")
	}
	csiMapper := mounter.(*csiBlockMapper)

	// validate mounter fields
	if string(csiMapper.driverName) != testDriver {
		t.Error("CSI block mapper missing driver name")
	}
	if csiMapper.volumeID != testVol {
		t.Error("CSI block mapper missing volumeID")
	}

	if csiMapper.podUID == types.UID("") {
		t.Error("CSI block mapper missing pod.UID")
	}
	csiClient, err := csiMapper.csiClientGetter.Get()
	if csiClient == nil {
		t.Error("mapper csiClient is nil")
	}

	// ensure data file is created
	dataFile := getVolumeDeviceDataDir(csiMapper.spec.Name(), plug.host)
	if _, err := os.Stat(dataFile); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("data file not created %s", dataFile)
		} else {
			t.Fatal(err)
		}
	}
}

func TestPluginNewUnmapper(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	registerFakePlugin(testDriver, "endpoint", []string{"1.0.0"}, t)
	pv := makeTestPV("test-pv", 10, testDriver, testVol)

	// save the data file to re-create client
	dir := getVolumeDeviceDataDir(pv.ObjectMeta.Name, plug.host)
	if err := os.MkdirAll(dir, 0755); err != nil && !os.IsNotExist(err) {
		t.Errorf("failed to create dir [%s]: %v", dir, err)
	}

	if err := saveVolumeData(
		dir,
		volDataFileName,
		map[string]string{
			volDataKey.specVolID:  pv.ObjectMeta.Name,
			volDataKey.driverName: testDriver,
			volDataKey.volHandle:  testVol,
		},
	); err != nil {
		t.Fatalf("failed to save volume data: %v", err)
	}

	// test unmounter
	unmapper, err := plug.NewBlockVolumeUnmapper(pv.ObjectMeta.Name, testPodUID)
	csiUnmapper := unmapper.(*csiBlockMapper)

	if err != nil {
		t.Fatalf("Failed to make a new Unmounter: %v", err)
	}

	if csiUnmapper == nil {
		t.Fatal("failed to create CSI Unmounter")
	}

	if csiUnmapper.podUID != testPodUID {
		t.Error("podUID not set")
	}

	if csiUnmapper.specName != pv.ObjectMeta.Name {
		t.Error("specName not set")
	}

	csiClient, err := csiUnmapper.csiClientGetter.Get()
	if csiClient == nil {
		t.Error("unmapper csiClient is nil")
	}

	// test loaded vol data
	if string(csiUnmapper.driverName) != testDriver {
		t.Error("unmapper driverName not set")
	}
	if csiUnmapper.volumeID != testVol {
		t.Error("unmapper volumeHandle not set")
	}
}

func TestPluginConstructBlockVolumeSpec(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
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
		deviceDataDir := getVolumeDeviceDataDir(tc.specVolID, plug.host)

		// create data file in csi plugin dir
		if tc.data != nil {
			if err := os.MkdirAll(deviceDataDir, 0755); err != nil && !os.IsNotExist(err) {
				t.Errorf("failed to create dir [%s]: %v", deviceDataDir, err)
			}
			if err := saveVolumeData(deviceDataDir, volDataFileName, tc.data); err != nil {
				t.Fatal(err)
			}
		}

		// rebuild spec
		spec, err := plug.ConstructBlockVolumeSpec("test-podUID", tc.specVolID, getVolumeDevicePluginDir(tc.specVolID, plug.host))
		if tc.shouldFail {
			if err == nil {
				t.Fatal("expecting ConstructVolumeSpec to fail, but got nil error")
			}
			continue
		}

		if spec.PersistentVolume.Spec.VolumeMode == nil {
			t.Fatalf("Volume mode has not been set.")
		}

		if *spec.PersistentVolume.Spec.VolumeMode != api.PersistentVolumeBlock {
			t.Errorf("Unexpected volume mode %q", *spec.PersistentVolume.Spec.VolumeMode)
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

func TestValidatePlugin(t *testing.T) {
	testCases := []struct {
		pluginName           string
		endpoint             string
		versions             []string
		foundInDeprecatedDir bool
		shouldFail           bool
	}{
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:             []string{"v1.0.0"},
			foundInDeprecatedDir: false,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:             []string{"0.3.0"},
			foundInDeprecatedDir: false,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:             []string{"0.2.0"},
			foundInDeprecatedDir: false,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions:             []string{"v1.0.0"},
			foundInDeprecatedDir: true,
			shouldFail:           true,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions:             []string{"v0.3.0"},
			foundInDeprecatedDir: true,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions:             []string{"0.2.0"},
			foundInDeprecatedDir: true,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:             []string{"0.2.0", "v0.3.0"},
			foundInDeprecatedDir: false,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions:             []string{"0.2.0", "v0.3.0"},
			foundInDeprecatedDir: true,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:             []string{"0.2.0", "v1.0.0"},
			foundInDeprecatedDir: false,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions:             []string{"0.2.0", "v1.0.0"},
			foundInDeprecatedDir: true,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:             []string{"0.2.0", "v1.2.3"},
			foundInDeprecatedDir: false,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions:             []string{"0.2.0", "v1.2.3"},
			foundInDeprecatedDir: true,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:             []string{"v1.2.3", "v0.3.0"},
			foundInDeprecatedDir: false,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions:             []string{"v1.2.3", "v0.3.0"},
			foundInDeprecatedDir: true,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:             []string{"v1.2.3", "v0.3.0", "2.0.1"},
			foundInDeprecatedDir: false,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions:             []string{"v1.2.3", "v0.3.0", "2.0.1"},
			foundInDeprecatedDir: true,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions:             []string{"v0.3.0", "2.0.1"},
			foundInDeprecatedDir: true,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:             []string{"v1.2.3", "4.9.12", "v0.3.0", "2.0.1"},
			foundInDeprecatedDir: false,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions:             []string{"v1.2.3", "4.9.12", "v0.3.0", "2.0.1"},
			foundInDeprecatedDir: true,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:             []string{"v1.2.3", "boo", "v0.3.0", "2.0.1"},
			foundInDeprecatedDir: false,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions:             []string{"v1.2.3", "boo", "v0.3.0", "2.0.1"},
			foundInDeprecatedDir: true,
			shouldFail:           false,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:             []string{"4.9.12", "2.0.1"},
			foundInDeprecatedDir: false,
			shouldFail:           true,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions:             []string{"4.9.12", "2.0.1"},
			foundInDeprecatedDir: true,
			shouldFail:           true,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:             []string{},
			foundInDeprecatedDir: false,
			shouldFail:           true,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions:             []string{},
			foundInDeprecatedDir: true,
			shouldFail:           true,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:             []string{"var", "boo", "foo"},
			foundInDeprecatedDir: false,
			shouldFail:           true,
		},
		{
			pluginName:           "test.plugin",
			endpoint:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions:             []string{"var", "boo", "foo"},
			foundInDeprecatedDir: true,
			shouldFail:           true,
		},
	}

	for _, tc := range testCases {
		// Arrange & Act
		err := PluginHandler.ValidatePlugin(tc.pluginName, tc.endpoint, tc.versions, tc.foundInDeprecatedDir)

		// Assert
		if tc.shouldFail && err == nil {
			t.Fatalf("expecting ValidatePlugin to fail, but got nil error for testcase: %#v", tc)
		}
		if !tc.shouldFail && err != nil {
			t.Fatalf("unexpected error during ValidatePlugin for testcase: %#v\r\n err:%v", tc, err)
		}
	}
}

func TestValidatePluginExistingDriver(t *testing.T) {
	testCases := []struct {
		pluginName1           string
		endpoint1             string
		versions1             []string
		pluginName2           string
		endpoint2             string
		versions2             []string
		foundInDeprecatedDir2 bool
		shouldFail            bool
	}{
		{
			pluginName1:           "test.plugin",
			endpoint1:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions1:             []string{"v1.0.0"},
			pluginName2:           "test.plugin2",
			endpoint2:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions2:             []string{"v1.0.0"},
			foundInDeprecatedDir2: false,
			shouldFail:            false,
		},
		{
			pluginName1:           "test.plugin",
			endpoint1:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions1:             []string{"v1.0.0"},
			pluginName2:           "test.plugin2",
			endpoint2:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions2:             []string{"v1.0.0"},
			foundInDeprecatedDir2: true,
			shouldFail:            true,
		},
		{
			pluginName1:           "test.plugin",
			endpoint1:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions1:             []string{"v1.0.0"},
			pluginName2:           "test.plugin",
			endpoint2:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions2:             []string{"v1.0.0"},
			foundInDeprecatedDir2: false,
			shouldFail:            true,
		},
		{
			pluginName1:           "test.plugin",
			endpoint1:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions1:             []string{"v1.0.0"},
			pluginName2:           "test.plugin",
			endpoint2:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions2:             []string{"v1.0.0"},
			foundInDeprecatedDir2: false,
			shouldFail:            true,
		},
		{
			pluginName1:           "test.plugin",
			endpoint1:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions1:             []string{"v1.0.0"},
			pluginName2:           "test.plugin",
			endpoint2:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions2:             []string{"v1.0.0"},
			foundInDeprecatedDir2: true,
			shouldFail:            true,
		},
		{
			pluginName1:           "test.plugin",
			endpoint1:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions1:             []string{"v0.3.0", "0.2.0"},
			pluginName2:           "test.plugin",
			endpoint2:             "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions2:             []string{"1.0.0"},
			foundInDeprecatedDir2: false,
			shouldFail:            false,
		},
		{
			pluginName1:           "test.plugin",
			endpoint1:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions1:             []string{"v0.3.0", "0.2.0"},
			pluginName2:           "test.plugin",
			endpoint2:             "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions2:             []string{"1.0.0"},
			foundInDeprecatedDir2: true,
			shouldFail:            true,
		},
	}

	for _, tc := range testCases {
		// Arrange & Act
		csiDrivers = csiDriversStore{driversMap: map[string]csiDriver{}}
		highestSupportedVersions1, err := highestSupportedVersion(tc.versions1)
		if err != nil {
			t.Fatalf("unexpected error parsing version for testcase: %#v", tc)
		}

		csiDrivers.driversMap[tc.pluginName1] = csiDriver{driverName: tc.pluginName1, driverEndpoint: tc.endpoint1, highestSupportedVersion: highestSupportedVersions1}

		// Arrange & Act
		err = PluginHandler.ValidatePlugin(tc.pluginName2, tc.endpoint2, tc.versions2, tc.foundInDeprecatedDir2)

		// Assert
		if tc.shouldFail && err == nil {
			t.Fatalf("expecting ValidatePlugin to fail, but got nil error for testcase: %#v", tc)
		}
		if !tc.shouldFail && err != nil {
			t.Fatalf("unexpected error during ValidatePlugin for testcase: %#v\r\n err:%v", tc, err)
		}
	}
}

func TestHighestSupportedVersion(t *testing.T) {
	testCases := []struct {
		versions                        []string
		expectedHighestSupportedVersion string
		shouldFail                      bool
	}{
		{
			versions:                        []string{"v1.0.0"},
			expectedHighestSupportedVersion: "1.0.0",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"0.3.0"},
			expectedHighestSupportedVersion: "0.3.0",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"0.2.0"},
			expectedHighestSupportedVersion: "0.2.0",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"1.0.0"},
			expectedHighestSupportedVersion: "1.0.0",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"v0.3.0"},
			expectedHighestSupportedVersion: "0.3.0",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"0.2.0"},
			expectedHighestSupportedVersion: "0.2.0",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"0.2.0", "v0.3.0"},
			expectedHighestSupportedVersion: "0.3.0",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"0.2.0", "v1.0.0"},
			expectedHighestSupportedVersion: "1.0.0",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"0.2.0", "v1.2.3"},
			expectedHighestSupportedVersion: "1.2.3",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"v1.2.3", "v0.3.0"},
			expectedHighestSupportedVersion: "1.2.3",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"v1.2.3", "v0.3.0", "2.0.1"},
			expectedHighestSupportedVersion: "1.2.3",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"v1.2.3", "4.9.12", "v0.3.0", "2.0.1"},
			expectedHighestSupportedVersion: "1.2.3",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"4.9.12", "2.0.1"},
			expectedHighestSupportedVersion: "",
			shouldFail:                      true,
		},
		{
			versions:                        []string{"v1.2.3", "boo", "v0.3.0", "2.0.1"},
			expectedHighestSupportedVersion: "1.2.3",
			shouldFail:                      false,
		},
		{
			versions:                        []string{},
			expectedHighestSupportedVersion: "",
			shouldFail:                      true,
		},
		{
			versions:                        []string{"var", "boo", "foo"},
			expectedHighestSupportedVersion: "",
			shouldFail:                      true,
		},
	}

	for _, tc := range testCases {
		// Arrange & Act
		actual, err := highestSupportedVersion(tc.versions)

		// Assert
		if tc.shouldFail && err == nil {
			t.Fatalf("expecting highestSupportedVersion to fail, but got nil error for testcase: %#v", tc)
		}
		if !tc.shouldFail && err != nil {
			t.Fatalf("unexpected error during ValidatePlugin for testcase: %#v\r\n err:%v", tc, err)
		}
		if tc.expectedHighestSupportedVersion != "" {
			result, err := actual.Compare(tc.expectedHighestSupportedVersion)
			if err != nil {
				t.Fatalf("comparison failed with %v for testcase %#v", err, tc)
			}
			if result != 0 {
				t.Fatalf("expectedHighestSupportedVersion %v, but got %v for tc: %#v", tc.expectedHighestSupportedVersion, actual, tc)
			}
		}
	}
}
