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
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	api "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

// create a plugin mgr to load plugins and setup a fake client
func newTestPlugin(t *testing.T, client *fakeclient.Clientset) (*csiPlugin, string) {
	tmpDir, err := utiltesting.MkTmpdir("csi-test")
	if err != nil {
		t.Fatalf("can't create temp dir: %v", err)
	}

	if client == nil {
		client = fakeclient.NewSimpleClientset()
	}

	client.Tracker().Add(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "fakeNode",
		},
		Spec: v1.NodeSpec{},
	})

	// Start informer for CSIDrivers.
	factory := informers.NewSharedInformerFactory(client, CsiResyncPeriod)
	csiDriverInformer := factory.Storage().V1().CSIDrivers()
	csiDriverLister := csiDriverInformer.Lister()
	go factory.Start(wait.NeverStop)

	host := volumetest.NewFakeVolumeHostWithCSINodeName(t,
		tmpDir,
		client,
		ProbeVolumePlugins(),
		"fakeNode",
		csiDriverLister,
	)

	pluginMgr := host.GetPluginMgr()
	plug, err := pluginMgr.FindPluginByName(CSIPluginName)
	if err != nil {
		t.Fatalf("can't find plugin %v", CSIPluginName)
	}

	csiPlug, ok := plug.(*csiPlugin)
	if !ok {
		t.Fatalf("cannot assert plugin to be type csiPlugin")
	}

	// Wait until the informer in CSI volume plugin has all CSIDrivers.
	wait.PollImmediate(TestInformerSyncPeriod, TestInformerSyncTimeout, func() (bool, error) {
		return csiDriverInformer.Informer().HasSynced(), nil
	})

	return csiPlug, tmpDir
}

func registerFakePlugin(pluginName, endpoint string, versions []string, t *testing.T) {
	highestSupportedVersions, err := highestSupportedVersion(versions)
	if err != nil {
		t.Fatalf("unexpected error parsing versions (%v) for pluginName %q endpoint %q: %#v", versions, pluginName, endpoint, err)
	}

	csiDrivers.Clear()
	csiDrivers.Set(pluginName, Driver{
		endpoint:                endpoint,
		highestSupportedVersion: highestSupportedVersions,
	})
}

func TestPluginGetPluginName(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil)
	defer os.RemoveAll(tmpDir)
	if plug.GetPluginName() != "kubernetes.io/csi" {
		t.Errorf("unexpected plugin name %v", plug.GetPluginName())
	}
}

func TestPluginGetVolumeName(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil)
	defer os.RemoveAll(tmpDir)
	testCases := []struct {
		name       string
		driverName string
		volName    string
		spec       *volume.Spec
		shouldFail bool
	}{
		{
			name:       "alphanum names",
			driverName: "testdr",
			volName:    "testvol",
			spec:       volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "testdr", "testvol"), false),
		},
		{
			name:       "mixchar driver",
			driverName: "test.dr.cc",
			volName:    "testvol",
			spec:       volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "test.dr.cc", "testvol"), false),
		},
		{
			name:       "mixchar volume",
			driverName: "testdr",
			volName:    "test-vol-name",
			spec:       volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "testdr", "test-vol-name"), false),
		},
		{
			name:       "mixchars all",
			driverName: "test-driver",
			volName:    "test.vol.name",
			spec:       volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "test-driver", "test.vol.name"), false),
		},
		{
			name:       "volume source with mixchars all",
			driverName: "test-driver",
			volName:    "test.vol.name",
			spec:       volume.NewSpecFromVolume(makeTestVol("test-pv", "test-driver")),
			shouldFail: true, // csi inline feature off
		},
		{
			name:       "missing spec",
			shouldFail: true,
		},
	}

	for _, tc := range testCases {
		t.Logf("testing: %s", tc.name)
		registerFakePlugin(tc.driverName, "endpoint", []string{"1.3.0"}, t)
		name, err := plug.GetVolumeName(tc.spec)
		if tc.shouldFail != (err != nil) {
			t.Fatal("shouldFail does match expected error")
		}
		if tc.shouldFail && err != nil {
			t.Log(err)
			continue
		}
		if name != fmt.Sprintf("%s%s%s", tc.driverName, volNameSep, tc.volName) {
			t.Errorf("unexpected volume name %s", name)
		}
	}
}

func TestPluginGetVolumeNameWithInline(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, true)()

	modes := []storagev1.VolumeLifecycleMode{
		storagev1.VolumeLifecyclePersistent,
	}
	driver := getTestCSIDriver(testDriver, nil, nil, modes)
	client := fakeclient.NewSimpleClientset(driver)
	plug, tmpDir := newTestPlugin(t, client)
	defer os.RemoveAll(tmpDir)
	testCases := []struct {
		name       string
		driverName string
		volName    string
		shouldFail bool
		spec       *volume.Spec
	}{
		{
			name:       "missing spec",
			shouldFail: true,
		},
		{
			name:       "alphanum names for pv",
			driverName: "testdr",
			volName:    "testvol",
			spec:       volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "testdr", "testvol"), false),
		},
		{
			name:       "alphanum names for vol source",
			driverName: "testdr",
			volName:    "testvol",
			spec:       volume.NewSpecFromVolume(makeTestVol("test-pv", "testdr")),
			shouldFail: true,
		},
	}

	for _, tc := range testCases {
		t.Logf("testing: %s", tc.name)
		registerFakePlugin(tc.driverName, "endpoint", []string{"1.3.0"}, t)
		name, err := plug.GetVolumeName(tc.spec)
		if tc.shouldFail != (err != nil) {
			t.Fatal("shouldFail does match expected error")
		}
		if tc.shouldFail && err != nil {
			t.Log(err)
			continue
		}
		if name != fmt.Sprintf("%s%s%s", tc.driverName, volNameSep, tc.volName) {
			t.Errorf("unexpected volume name %s", name)
		}
	}
}

func TestPluginCanSupport(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, false)()

	tests := []struct {
		name       string
		spec       *volume.Spec
		canSupport bool
	}{
		{
			name:       "no spec provided",
			canSupport: false,
		},
		{
			name:       "can support volume source",
			spec:       volume.NewSpecFromVolume(makeTestVol("test-vol", testDriver)),
			canSupport: false, // csi inline not enabled
		},
		{
			name:       "can support persistent volume source",
			spec:       volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 20, testDriver, testVol), true),
			canSupport: true,
		},
	}

	plug, tmpDir := newTestPlugin(t, nil)
	defer os.RemoveAll(tmpDir)
	registerFakePlugin(testDriver, "endpoint", []string{"1.0.0"}, t)

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			actual := plug.CanSupport(tc.spec)
			if tc.canSupport != actual {
				t.Errorf("expecting canSupport %t, got %t", tc.canSupport, actual)
			}
		})
	}
}

func TestPluginCanSupportWithInline(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, true)()

	tests := []struct {
		name       string
		spec       *volume.Spec
		canSupport bool
	}{
		{
			name:       "no spec provided",
			canSupport: false,
		},
		{
			name:       "can support volume source",
			spec:       volume.NewSpecFromVolume(makeTestVol("test-vol", testDriver)),
			canSupport: true,
		},
		{
			name:       "can support persistent volume source",
			spec:       volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 20, testDriver, testVol), true),
			canSupport: true,
		},
	}

	plug, tmpDir := newTestPlugin(t, nil)
	defer os.RemoveAll(tmpDir)
	registerFakePlugin(testDriver, "endpoint", []string{"1.0.0"}, t)

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			actual := plug.CanSupport(tc.spec)
			if tc.canSupport != actual {
				t.Errorf("expecting canSupport %t, got %t", tc.canSupport, actual)
			}
		})
	}
}

func TestPluginConstructVolumeSpec(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil)
	defer os.RemoveAll(tmpDir)

	testCases := []struct {
		name       string
		originSpec *volume.Spec
		specVolID  string
		volHandle  string
		podUID     types.UID
		shouldFail bool
	}{
		{
			name:       "construct spec1 from original persistent spec",
			specVolID:  "test.vol.id",
			volHandle:  "testvol-handle1",
			originSpec: volume.NewSpecFromPersistentVolume(makeTestPV("test.vol.id", 20, testDriver, "testvol-handle1"), true),
			podUID:     types.UID(fmt.Sprintf("%08X", rand.Uint64())),
		},
		{
			name:       "construct spec2 from original persistent spec",
			specVolID:  "spec2",
			volHandle:  "handle2",
			originSpec: volume.NewSpecFromPersistentVolume(makeTestPV("spec2", 20, testDriver, "handle2"), true),
			podUID:     types.UID(fmt.Sprintf("%08X", rand.Uint64())),
		},
		{
			name:       "construct spec from original volume spec",
			specVolID:  "volspec",
			originSpec: volume.NewSpecFromVolume(makeTestVol("spec2", testDriver)),
			podUID:     types.UID(fmt.Sprintf("%08X", rand.Uint64())),
			shouldFail: true, // csi inline off
		},
	}

	registerFakePlugin(testDriver, "endpoint", []string{"1.0.0"}, t)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mounter, err := plug.NewMounter(
				tc.originSpec,
				&api.Pod{ObjectMeta: meta.ObjectMeta{UID: tc.podUID, Namespace: testns}},
				volume.VolumeOptions{},
			)
			if tc.shouldFail && err != nil {
				t.Log(err)
				return
			}
			if !tc.shouldFail && err != nil {
				t.Fatal(err)
			}
			if mounter == nil {
				t.Fatal("failed to create CSI mounter")
			}
			csiMounter := mounter.(*csiMountMgr)

			// rebuild spec
			spec, err := plug.ConstructVolumeSpec("test-pv", filepath.Dir(csiMounter.GetPath()))
			if err != nil {
				t.Fatal(err)
			}
			if spec == nil {
				t.Fatal("nil volume.Spec constructed")
			}

			// inspect spec
			if spec.PersistentVolume == nil || spec.PersistentVolume.Spec.CSI == nil {
				t.Fatal("CSIPersistentVolume not found in constructed spec ")
			}

			volHandle := spec.PersistentVolume.Spec.CSI.VolumeHandle
			if volHandle != tc.originSpec.PersistentVolume.Spec.CSI.VolumeHandle {
				t.Error("unexpected volumeHandle constructed:", volHandle)
			}
			driverName := spec.PersistentVolume.Spec.CSI.Driver
			if driverName != tc.originSpec.PersistentVolume.Spec.CSI.Driver {
				t.Error("unexpected driverName constructed:", driverName)
			}

			if spec.PersistentVolume.Spec.VolumeMode == nil {
				t.Fatalf("Volume mode has not been set.")
			}

			if *spec.PersistentVolume.Spec.VolumeMode != api.PersistentVolumeFilesystem {
				t.Errorf("Unexpected volume mode %q", *spec.PersistentVolume.Spec.VolumeMode)
			}

			if spec.Name() != tc.specVolID {
				t.Errorf("Unexpected spec name constructed %s", spec.Name())
			}

		})
	}
}

func TestPluginConstructVolumeSpecWithInline(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, true)()

	testCases := []struct {
		name       string
		originSpec *volume.Spec
		specVolID  string
		volHandle  string
		podUID     types.UID
		shouldFail bool
		modes      []storagev1.VolumeLifecycleMode
	}{
		{
			name:       "construct spec1 from persistent spec",
			specVolID:  "test.vol.id",
			volHandle:  "testvol-handle1",
			originSpec: volume.NewSpecFromPersistentVolume(makeTestPV("test.vol.id", 20, testDriver, "testvol-handle1"), true),
			podUID:     types.UID(fmt.Sprintf("%08X", rand.Uint64())),
			modes:      []storagev1.VolumeLifecycleMode{storagev1.VolumeLifecyclePersistent},
		},
		{
			name:       "construct spec2 from persistent spec",
			specVolID:  "spec2",
			volHandle:  "handle2",
			originSpec: volume.NewSpecFromPersistentVolume(makeTestPV("spec2", 20, testDriver, "handle2"), true),
			podUID:     types.UID(fmt.Sprintf("%08X", rand.Uint64())),
			modes:      []storagev1.VolumeLifecycleMode{storagev1.VolumeLifecyclePersistent},
		},
		{
			name:       "construct spec2 from persistent spec, missing mode",
			specVolID:  "spec2",
			volHandle:  "handle2",
			originSpec: volume.NewSpecFromPersistentVolume(makeTestPV("spec2", 20, testDriver, "handle2"), true),
			podUID:     types.UID(fmt.Sprintf("%08X", rand.Uint64())),
			modes:      []storagev1.VolumeLifecycleMode{},
			shouldFail: true,
		},
		{
			name:       "construct spec from volume spec",
			specVolID:  "volspec",
			originSpec: volume.NewSpecFromVolume(makeTestVol("volspec", testDriver)),
			podUID:     types.UID(fmt.Sprintf("%08X", rand.Uint64())),
			modes:      []storagev1.VolumeLifecycleMode{storagev1.VolumeLifecycleEphemeral},
		},
		{
			name:       "construct spec from volume spec2",
			specVolID:  "volspec2",
			originSpec: volume.NewSpecFromVolume(makeTestVol("volspec2", testDriver)),
			podUID:     types.UID(fmt.Sprintf("%08X", rand.Uint64())),
			modes:      []storagev1.VolumeLifecycleMode{storagev1.VolumeLifecycleEphemeral},
		},
		{
			name:       "construct spec from volume spec2, missing mode",
			specVolID:  "volspec2",
			originSpec: volume.NewSpecFromVolume(makeTestVol("volspec2", testDriver)),
			podUID:     types.UID(fmt.Sprintf("%08X", rand.Uint64())),
			modes:      []storagev1.VolumeLifecycleMode{},
			shouldFail: true,
		},
		{
			name:       "missing spec",
			podUID:     types.UID(fmt.Sprintf("%08X", rand.Uint64())),
			shouldFail: true,
		},
	}

	registerFakePlugin(testDriver, "endpoint", []string{"1.0.0"}, t)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			driver := getTestCSIDriver(testDriver, nil, nil, tc.modes)
			client := fakeclient.NewSimpleClientset(driver)
			plug, tmpDir := newTestPlugin(t, client)
			defer os.RemoveAll(tmpDir)

			mounter, err := plug.NewMounter(
				tc.originSpec,
				&api.Pod{ObjectMeta: meta.ObjectMeta{UID: tc.podUID, Namespace: testns}},
				volume.VolumeOptions{},
			)
			if tc.shouldFail && err != nil {
				t.Log(err)
				return
			}
			if !tc.shouldFail && err != nil {
				t.Fatal(err)
			}
			if mounter == nil {
				t.Fatal("failed to create CSI mounter")
			}
			csiMounter := mounter.(*csiMountMgr)

			// rebuild spec
			spec, err := plug.ConstructVolumeSpec("test-pv", filepath.Dir(csiMounter.GetPath()))
			if err != nil {
				t.Fatal(err)
			}
			if spec == nil {
				t.Fatal("nil volume.Spec constructed")
			}

			if spec.Name() != tc.specVolID {
				t.Errorf("unexpected spec name constructed volume.Spec: %s", spec.Name())
			}

			switch {
			case spec.Volume != nil:
				if spec.Volume.CSI == nil {
					t.Error("missing CSIVolumeSource in constructed volume.Spec")
				}
				if spec.Volume.CSI.Driver != tc.originSpec.Volume.CSI.Driver {
					t.Error("unexpected driver in constructed volume source:", spec.Volume.CSI.Driver)
				}

			case spec.PersistentVolume != nil:
				if spec.PersistentVolume.Spec.CSI == nil {
					t.Fatal("missing CSIPersistentVolumeSource in constructed volume.spec")
				}
				volHandle := spec.PersistentVolume.Spec.CSI.VolumeHandle
				if volHandle != tc.originSpec.PersistentVolume.Spec.CSI.VolumeHandle {
					t.Error("unexpected volumeHandle constructed in persistent volume source:", volHandle)
				}
				driverName := spec.PersistentVolume.Spec.CSI.Driver
				if driverName != tc.originSpec.PersistentVolume.Spec.CSI.Driver {
					t.Error("unexpected driverName constructed in persistent volume source:", driverName)
				}
				if spec.PersistentVolume.Spec.VolumeMode == nil {
					t.Fatalf("Volume mode has not been set.")
				}
				if *spec.PersistentVolume.Spec.VolumeMode != api.PersistentVolumeFilesystem {
					t.Errorf("Unexpected volume mode %q", *spec.PersistentVolume.Spec.VolumeMode)
				}
			default:
				t.Fatal("invalid volume.Spec constructed")
			}

		})
	}
}

func TestPluginNewMounter(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	tests := []struct {
		name                string
		spec                *volume.Spec
		podUID              types.UID
		namespace           string
		volumeLifecycleMode storagev1.VolumeLifecycleMode
		shouldFail          bool
	}{
		{
			name:                "mounter from persistent volume source",
			spec:                volume.NewSpecFromPersistentVolume(makeTestPV("test-pv1", 20, testDriver, testVol), true),
			podUID:              types.UID(fmt.Sprintf("%08X", rand.Uint64())),
			namespace:           "test-ns1",
			volumeLifecycleMode: storagev1.VolumeLifecyclePersistent,
		},
		{
			name:                "mounter from volume source",
			spec:                volume.NewSpecFromVolume(makeTestVol("test-vol1", testDriver)),
			podUID:              types.UID(fmt.Sprintf("%08X", rand.Uint64())),
			namespace:           "test-ns2",
			volumeLifecycleMode: storagev1.VolumeLifecycleEphemeral,
			shouldFail:          true, // csi inline not enabled
		},
		{
			name:       "mounter from no spec provided",
			shouldFail: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plug, tmpDir := newTestPlugin(t, nil)
			defer os.RemoveAll(tmpDir)

			registerFakePlugin(testDriver, "endpoint", []string{"1.2.0"}, t)
			mounter, err := plug.NewMounter(
				test.spec,
				&api.Pod{ObjectMeta: meta.ObjectMeta{UID: test.podUID, Namespace: test.namespace}},
				volume.VolumeOptions{},
			)
			if test.shouldFail != (err != nil) {
				t.Fatal("Unexpected error:", err)
			}
			if test.shouldFail && err != nil {
				t.Log(err)
				return
			}

			if mounter == nil {
				t.Fatal("failed to create CSI mounter")
			}
			csiMounter := mounter.(*csiMountMgr)

			// validate mounter fields
			if string(csiMounter.driverName) != testDriver {
				t.Error("mounter driver name not set")
			}
			if csiMounter.volumeID == "" {
				t.Error("mounter volume id not set")
			}
			if csiMounter.pod == nil {
				t.Error("mounter pod not set")
			}
			if string(csiMounter.podUID) != string(test.podUID) {
				t.Error("mounter podUID not set")
			}
			csiClient, err := csiMounter.csiClientGetter.Get()
			if csiClient == nil {
				t.Errorf("mounter csiClient is nil: %v", err)
			}
			if err != nil {
				t.Fatal(err)
			}
			if csiMounter.volumeLifecycleMode != test.volumeLifecycleMode {
				t.Error("unexpected driver mode:", csiMounter.volumeLifecycleMode)
			}

			// ensure data file is created
			dataDir := filepath.Dir(mounter.GetPath())
			dataFile := filepath.Join(dataDir, volDataFileName)
			if _, err := os.Stat(dataFile); err != nil {
				if os.IsNotExist(err) {
					t.Errorf("data file not created %s", dataFile)
				} else {
					t.Fatal(err)
				}
			}
			data, err := loadVolumeData(dataDir, volDataFileName)
			if err != nil {
				t.Fatal(err)
			}
			if data[volDataKey.specVolID] != csiMounter.spec.Name() {
				t.Error("volume data file unexpected specVolID:", data[volDataKey.specVolID])
			}
			if data[volDataKey.volHandle] != csiMounter.volumeID {
				t.Error("volume data file unexpected volHandle:", data[volDataKey.volHandle])
			}
			if data[volDataKey.driverName] != string(csiMounter.driverName) {
				t.Error("volume data file unexpected driverName:", data[volDataKey.driverName])
			}
			if data[volDataKey.nodeName] != string(csiMounter.plugin.host.GetNodeName()) {
				t.Error("volume data file unexpected nodeName:", data[volDataKey.nodeName])
			}
			if data[volDataKey.volumeLifecycleMode] != string(test.volumeLifecycleMode) {
				t.Error("volume data file unexpected volumeLifecycleMode:", data[volDataKey.volumeLifecycleMode])
			}
		})
	}
}

func TestPluginNewMounterWithInline(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, true)()
	bothModes := []storagev1.VolumeLifecycleMode{
		storagev1.VolumeLifecycleEphemeral,
		storagev1.VolumeLifecyclePersistent,
	}
	persistentMode := []storagev1.VolumeLifecycleMode{
		storagev1.VolumeLifecyclePersistent,
	}
	ephemeralMode := []storagev1.VolumeLifecycleMode{
		storagev1.VolumeLifecycleEphemeral,
	}
	tests := []struct {
		name                string
		spec                *volume.Spec
		podUID              types.UID
		namespace           string
		volumeLifecycleMode storagev1.VolumeLifecycleMode
		shouldFail          bool
	}{
		{
			name:       "mounter with missing spec",
			shouldFail: true,
		},
		{
			name: "mounter with spec with both volSrc and pvSrc",
			spec: &volume.Spec{
				Volume:           makeTestVol("test-vol1", testDriver),
				PersistentVolume: makeTestPV("test-pv1", 20, testDriver, testVol),
				ReadOnly:         true,
			},
			shouldFail: true,
		},
		{
			name:                "mounter with persistent volume source",
			spec:                volume.NewSpecFromPersistentVolume(makeTestPV("test-pv1", 20, testDriver, testVol), true),
			podUID:              types.UID(fmt.Sprintf("%08X", rand.Uint64())),
			namespace:           "test-ns1",
			volumeLifecycleMode: storagev1.VolumeLifecyclePersistent,
		},
		{
			name:                "mounter with volume source",
			spec:                volume.NewSpecFromVolume(makeTestVol("test-vol1", testDriver)),
			podUID:              types.UID(fmt.Sprintf("%08X", rand.Uint64())),
			namespace:           "test-ns2",
			volumeLifecycleMode: storagev1.VolumeLifecycleEphemeral,
		},
	}

	runAll := func(t *testing.T, supported []storagev1.VolumeLifecycleMode) {
		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				driver := getTestCSIDriver(testDriver, nil, nil, supported)
				fakeClient := fakeclient.NewSimpleClientset(driver)
				plug, tmpDir := newTestPlugin(t, fakeClient)
				defer os.RemoveAll(tmpDir)

				registerFakePlugin(testDriver, "endpoint", []string{"1.2.0"}, t)

				mounter, err := plug.NewMounter(
					test.spec,
					&api.Pod{ObjectMeta: meta.ObjectMeta{UID: test.podUID, Namespace: test.namespace}},
					volume.VolumeOptions{},
				)

				// Some test cases are meant to fail because their input data is broken.
				shouldFail := test.shouldFail
				// Others fail if the driver does not support the volume mode.
				if !containsVolumeMode(supported, test.volumeLifecycleMode) {
					shouldFail = true
				}
				if shouldFail != (err != nil) {
					t.Fatal("Unexpected error:", err)
				}
				if shouldFail && err != nil {
					t.Log(err)
					return
				}

				if mounter == nil {
					t.Fatal("failed to create CSI mounter")
				}
				csiMounter := mounter.(*csiMountMgr)

				// validate mounter fields
				if string(csiMounter.driverName) != testDriver {
					t.Error("mounter driver name not set")
				}
				if csiMounter.volumeID == "" {
					t.Error("mounter volume id not set")
				}
				if csiMounter.pod == nil {
					t.Error("mounter pod not set")
				}
				if string(csiMounter.podUID) != string(test.podUID) {
					t.Error("mounter podUID not set")
				}
				csiClient, err := csiMounter.csiClientGetter.Get()
				if csiClient == nil {
					t.Errorf("mounter csiClient is nil: %v", err)
				}
				if csiMounter.volumeLifecycleMode != test.volumeLifecycleMode {
					t.Error("unexpected driver mode:", csiMounter.volumeLifecycleMode)
				}

				// ensure data file is created
				dataDir := filepath.Dir(mounter.GetPath())
				dataFile := filepath.Join(dataDir, volDataFileName)
				if _, err := os.Stat(dataFile); err != nil {
					if os.IsNotExist(err) {
						t.Errorf("data file not created %s", dataFile)
					} else {
						t.Fatal(err)
					}
				}
				data, err := loadVolumeData(dataDir, volDataFileName)
				if err != nil {
					t.Fatal(err)
				}
				if data[volDataKey.specVolID] != csiMounter.spec.Name() {
					t.Error("volume data file unexpected specVolID:", data[volDataKey.specVolID])
				}
				if data[volDataKey.volHandle] != csiMounter.volumeID {
					t.Error("volume data file unexpected volHandle:", data[volDataKey.volHandle])
				}
				if data[volDataKey.driverName] != string(csiMounter.driverName) {
					t.Error("volume data file unexpected driverName:", data[volDataKey.driverName])
				}
				if data[volDataKey.nodeName] != string(csiMounter.plugin.host.GetNodeName()) {
					t.Error("volume data file unexpected nodeName:", data[volDataKey.nodeName])
				}
				if data[volDataKey.volumeLifecycleMode] != string(csiMounter.volumeLifecycleMode) {
					t.Error("volume data file unexpected volumeLifecycleMode:", data[volDataKey.volumeLifecycleMode])
				}
			})
		}
	}

	t.Run("both supported", func(t *testing.T) {
		runAll(t, bothModes)
	})
	t.Run("persistent supported", func(t *testing.T) {
		runAll(t, persistentMode)
	})
	t.Run("ephemeral supported", func(t *testing.T) {
		runAll(t, ephemeralMode)
	})
}

func TestPluginNewUnmounter(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil)
	defer os.RemoveAll(tmpDir)

	registerFakePlugin(testDriver, "endpoint", []string{"1.0.0"}, t)
	pv := makeTestPV("test-pv", 10, testDriver, testVol)

	// save the data file to re-create client
	dir := filepath.Join(getTargetPath(testPodUID, pv.ObjectMeta.Name, plug.host), "/mount")
	if err := os.MkdirAll(dir, 0755); err != nil && !os.IsNotExist(err) {
		t.Errorf("failed to create dir [%s]: %v", dir, err)
	}

	if err := saveVolumeData(
		filepath.Dir(dir),
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
		t.Errorf("mounter csiClient is nil: %v", err)
	}
}

func TestPluginNewAttacher(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil)
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
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil)
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

func TestPluginCanAttach(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, true)()
	tests := []struct {
		name       string
		driverName string
		spec       *volume.Spec
		canAttach  bool
		shouldFail bool
	}{
		{
			name:       "non-attachable inline",
			driverName: "attachable-inline",
			spec:       volume.NewSpecFromVolume(makeTestVol("test-vol", "attachable-inline")),
			canAttach:  false,
		},
		{
			name:       "attachable PV",
			driverName: "attachable-pv",
			spec:       volume.NewSpecFromPersistentVolume(makeTestPV("test-vol", 20, "attachable-pv", testVol), true),
			canAttach:  true,
		},
		{
			name:       "incomplete spec",
			driverName: "attachable-pv",
			spec:       &volume.Spec{ReadOnly: true},
			canAttach:  false,
			shouldFail: true,
		},
		{
			name:       "nil spec",
			driverName: "attachable-pv",
			canAttach:  false,
			shouldFail: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			csiDriver := getTestCSIDriver(test.driverName, nil, &test.canAttach, nil)
			fakeCSIClient := fakeclient.NewSimpleClientset(csiDriver)
			plug, tmpDir := newTestPlugin(t, fakeCSIClient)
			defer os.RemoveAll(tmpDir)

			pluginCanAttach, err := plug.CanAttach(test.spec)
			if err != nil && !test.shouldFail {
				t.Fatalf("unexected plugin.CanAttach error: %s", err)
			}
			if pluginCanAttach != test.canAttach {
				t.Fatalf("expecting plugin.CanAttach %t got %t", test.canAttach, pluginCanAttach)
			}
		})
	}
}

func TestPluginFindAttachablePlugin(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, true)()
	tests := []struct {
		name       string
		driverName string
		spec       *volume.Spec
		canAttach  bool
		shouldFail bool
	}{
		{
			name:       "non-attachable inline",
			driverName: "attachable-inline",
			spec:       volume.NewSpecFromVolume(makeTestVol("test-vol", "attachable-inline")),
			canAttach:  false,
		},
		{
			name:       "attachable PV",
			driverName: "attachable-pv",
			spec:       volume.NewSpecFromPersistentVolume(makeTestPV("test-vol", 20, "attachable-pv", testVol), true),
			canAttach:  true,
		},
		{
			name:       "incomplete spec",
			driverName: "attachable-pv",
			spec:       &volume.Spec{ReadOnly: true},
			canAttach:  false,
			shouldFail: true,
		},
		{
			name:       "nil spec",
			driverName: "attachable-pv",
			canAttach:  false,
			shouldFail: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tmpDir, err := utiltesting.MkTmpdir("csi-test")
			if err != nil {
				t.Fatalf("can't create temp dir: %v", err)
			}
			defer os.RemoveAll(tmpDir)

			client := fakeclient.NewSimpleClientset(
				getTestCSIDriver(test.driverName, nil, &test.canAttach, nil),
				&v1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name: "fakeNode",
					},
					Spec: v1.NodeSpec{},
				},
			)
			factory := informers.NewSharedInformerFactory(client, CsiResyncPeriod)
			host := volumetest.NewFakeVolumeHostWithCSINodeName(t,
				tmpDir,
				client,
				ProbeVolumePlugins(),
				"fakeNode",
				factory.Storage().V1().CSIDrivers().Lister(),
			)

			plugMgr := host.GetPluginMgr()

			plugin, err := plugMgr.FindAttachablePluginBySpec(test.spec)
			if err != nil && !test.shouldFail {
				t.Fatalf("unexected error calling pluginMgr.FindAttachablePluginBySpec: %s", err)
			}
			if (plugin != nil) != test.canAttach {
				t.Fatal("expecting attachable plugin, but got nil")
			}
		})
	}
}

func TestPluginCanDeviceMount(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, true)()
	tests := []struct {
		name           string
		driverName     string
		spec           *volume.Spec
		canDeviceMount bool
		shouldFail     bool
	}{
		{
			name:           "non device mountable inline",
			driverName:     "inline-driver",
			spec:           volume.NewSpecFromVolume(makeTestVol("test-vol", "inline-driver")),
			canDeviceMount: false,
		},
		{
			name:           "device mountable PV",
			driverName:     "device-mountable-pv",
			spec:           volume.NewSpecFromPersistentVolume(makeTestPV("test-vol", 20, "device-mountable-pv", testVol), true),
			canDeviceMount: true,
		},
		{
			name:           "incomplete spec",
			driverName:     "device-unmountable",
			spec:           &volume.Spec{ReadOnly: true},
			canDeviceMount: false,
			shouldFail:     true,
		},
		{
			name:           "missing spec",
			driverName:     "device-unmountable",
			canDeviceMount: false,
			shouldFail:     true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plug, tmpDir := newTestPlugin(t, nil)
			defer os.RemoveAll(tmpDir)

			pluginCanDeviceMount, err := plug.CanDeviceMount(test.spec)
			if err != nil && !test.shouldFail {
				t.Fatalf("unexpected error in plug.CanDeviceMount: %s", err)
			}
			if pluginCanDeviceMount != test.canDeviceMount {
				t.Fatalf("expecting plugin.CanAttach %t got %t", test.canDeviceMount, pluginCanDeviceMount)
			}
		})
	}
}

func TestPluginFindDeviceMountablePluginBySpec(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, true)()
	tests := []struct {
		name           string
		driverName     string
		spec           *volume.Spec
		canDeviceMount bool
		shouldFail     bool
	}{
		{
			name:           "non device mountable inline",
			driverName:     "inline-driver",
			spec:           volume.NewSpecFromVolume(makeTestVol("test-vol", "inline-driver")),
			canDeviceMount: false,
		},
		{
			name:           "device mountable PV",
			driverName:     "device-mountable-pv",
			spec:           volume.NewSpecFromPersistentVolume(makeTestPV("test-vol", 20, "device-mountable-pv", testVol), true),
			canDeviceMount: true,
		},
		{
			name:           "incomplete spec",
			driverName:     "device-unmountable",
			spec:           &volume.Spec{ReadOnly: true},
			canDeviceMount: false,
			shouldFail:     true,
		},
		{
			name:           "missing spec",
			driverName:     "device-unmountable",
			canDeviceMount: false,
			shouldFail:     true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tmpDir, err := utiltesting.MkTmpdir("csi-test")
			if err != nil {
				t.Fatalf("can't create temp dir: %v", err)
			}
			defer os.RemoveAll(tmpDir)

			client := fakeclient.NewSimpleClientset(
				&v1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name: "fakeNode",
					},
					Spec: v1.NodeSpec{},
				},
			)
			host := volumetest.NewFakeVolumeHostWithCSINodeName(t, tmpDir, client, ProbeVolumePlugins(), "fakeNode", nil)
			plugMgr := host.GetPluginMgr()
			plug, err := plugMgr.FindDeviceMountablePluginBySpec(test.spec)
			if err != nil && !test.shouldFail {
				t.Fatalf("unexpected error in plugMgr.FindDeviceMountablePluginBySpec: %s", err)
			}
			if (plug != nil) != test.canDeviceMount {
				t.Fatalf("expecting deviceMountablePlugin, but got nil")
			}
		})
	}
}

func TestPluginNewBlockMapper(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil)
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
		t.Errorf("mapper csiClient is nil: %v", err)
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
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil)
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
		t.Errorf("unmapper csiClient is nil: %v", err)
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
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil)
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
		pluginName string
		endpoint   string
		versions   []string
		shouldFail bool
	}{
		{
			pluginName: "test.plugin",
			endpoint:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:   []string{"v1.0.0"},
			shouldFail: false,
		},
		{
			pluginName: "test.plugin",
			endpoint:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:   []string{"0.3.0"},
			shouldFail: true,
		},
		{
			pluginName: "test.plugin",
			endpoint:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:   []string{"0.2.0"},
			shouldFail: true,
		},
		{
			pluginName: "test.plugin",
			endpoint:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:   []string{"0.2.0", "v0.3.0"},
			shouldFail: true,
		},
		{
			pluginName: "test.plugin",
			endpoint:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:   []string{"0.2.0", "v1.0.0"},
			shouldFail: false,
		},
		{
			pluginName: "test.plugin",
			endpoint:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:   []string{"0.2.0", "v1.2.3"},
			shouldFail: false,
		},
		{
			pluginName: "test.plugin",
			endpoint:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:   []string{"v1.2.3", "v0.3.0"},
			shouldFail: false,
		},
		{
			pluginName: "test.plugin",
			endpoint:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:   []string{"v1.2.3", "v0.3.0", "2.0.1"},
			shouldFail: false,
		},
		{
			pluginName: "test.plugin",
			endpoint:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:   []string{"v1.2.3", "4.9.12", "v0.3.0", "2.0.1"},
			shouldFail: false,
		},
		{
			pluginName: "test.plugin",
			endpoint:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:   []string{"v1.2.3", "boo", "v0.3.0", "2.0.1"},
			shouldFail: false,
		},
		{
			pluginName: "test.plugin",
			endpoint:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:   []string{"4.9.12", "2.0.1"},
			shouldFail: true,
		},
		{
			pluginName: "test.plugin",
			endpoint:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:   []string{},
			shouldFail: true,
		},
		{
			pluginName: "test.plugin",
			endpoint:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions:   []string{"var", "boo", "foo"},
			shouldFail: true,
		},
	}

	for _, tc := range testCases {
		// Arrange & Act
		err := PluginHandler.ValidatePlugin(tc.pluginName, tc.endpoint, tc.versions)

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
		pluginName1 string
		endpoint1   string
		versions1   []string
		pluginName2 string
		endpoint2   string
		versions2   []string
		shouldFail  bool
	}{
		{
			pluginName1: "test.plugin",
			endpoint1:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions1:   []string{"v1.0.0"},
			pluginName2: "test.plugin2",
			endpoint2:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions2:   []string{"v1.0.0"},
			shouldFail:  false,
		},
		{
			pluginName1: "test.plugin",
			endpoint1:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions1:   []string{"v1.0.0"},
			pluginName2: "test.plugin",
			endpoint2:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions2:   []string{"v1.0.0"},
			shouldFail:  true,
		},
		{
			pluginName1: "test.plugin",
			endpoint1:   "/var/log/kubelet/plugins/myplugin/csi.sock",
			versions1:   []string{"v0.3.0", "v0.2.0", "v1.0.0"},
			pluginName2: "test.plugin",
			endpoint2:   "/var/log/kubelet/plugins_registry/myplugin/csi.sock",
			versions2:   []string{"v1.0.1"},
			shouldFail:  false,
		},
	}

	for _, tc := range testCases {
		// Arrange & Act
		highestSupportedVersions1, err := highestSupportedVersion(tc.versions1)
		if err != nil {
			t.Fatalf("unexpected error parsing version for testcase: %#v: %v", tc, err)
		}

		csiDrivers.Clear()
		csiDrivers.Set(tc.pluginName1, Driver{
			endpoint:                tc.endpoint1,
			highestSupportedVersion: highestSupportedVersions1,
		})

		// Arrange & Act
		err = PluginHandler.ValidatePlugin(tc.pluginName2, tc.endpoint2, tc.versions2)

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
			versions:   []string{"0.3.0"},
			shouldFail: true,
		},
		{
			versions:   []string{"0.2.0"},
			shouldFail: true,
		},
		{
			versions:                        []string{"1.0.0"},
			expectedHighestSupportedVersion: "1.0.0",
			shouldFail:                      false,
		},
		{
			versions:   []string{"v0.3.0"},
			shouldFail: true,
		},
		{
			versions:   []string{"0.2.0"},
			shouldFail: true,
		},
		{
			versions:   []string{"0.2.0", "v0.3.0"},
			shouldFail: true,
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
