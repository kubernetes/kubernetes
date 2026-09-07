/*
Copyright 2019 The Kubernetes Authors.

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
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	api "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/mount-utils"
)

// TestMain starting point for all tests.
// Surfaces klog flags by default to enable
// go test -v ./ --args <klog flags>
func TestMain(m *testing.M) {
	klog.InitFlags(flag.CommandLine)
	os.Exit(m.Run())
}

func makeTestPVWithMountOptions(name string, sizeGig int, driverName, volID string, mountOptions []string) *api.PersistentVolume {
	pv := makeTestPV(name, sizeGig, driverName, volID)
	pv.Spec.MountOptions = mountOptions
	return pv
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

func makeTestVol(name string, driverName string) *api.Volume {
	ro := false
	return &api.Volume{
		Name: name,
		VolumeSource: api.VolumeSource{
			CSI: &api.CSIVolumeSource{
				Driver:   driverName,
				ReadOnly: &ro,
			},
		},
	}
}

func getTestCSIDriver(name string, podInfoMount *bool, attachable *bool, volumeLifecycleModes []storagev1.VolumeLifecycleMode) *storagev1.CSIDriver {
	defaultFSGroupPolicy := storagev1.ReadWriteOnceWithFSTypeFSGroupPolicy
	seLinuxMountSupport := true
	noSElinuxMountSupport := false
	driver := &storagev1.CSIDriver{
		ObjectMeta: meta.ObjectMeta{
			Name: name,
		},
		Spec: storagev1.CSIDriverSpec{
			PodInfoOnMount:       podInfoMount,
			AttachRequired:       attachable,
			VolumeLifecycleModes: volumeLifecycleModes,
			FSGroupPolicy:        &defaultFSGroupPolicy,
		},
	}
	switch driver.Name {
	case "supports_selinux":
		driver.Spec.SELinuxMount = &seLinuxMountSupport
	case "no_selinux":
		driver.Spec.SELinuxMount = &noSElinuxMountSupport
	}
	return driver
}

func TestSaveVolumeData(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil)
	defer os.RemoveAll(tmpDir)
	testCases := []struct {
		name       string
		data       map[string]string
		shouldFail bool
	}{
		{name: "test with data ok", data: map[string]string{"key0": "val0", "_key1": "val1", "key2": "val2"}},
		{name: "test with data ok 2 ", data: map[string]string{"_key0_": "val0", "&key1": "val1", "key2": "val2"}},
	}

	for i, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		specVolID := fmt.Sprintf("spec-volid-%d", i)
		targetPath := getTargetPath(testPodUID, specVolID, plug.host)
		mountDir := filepath.Join(targetPath, "mount")
		if err := os.MkdirAll(mountDir, 0755); err != nil && !os.IsNotExist(err) {
			t.Errorf("failed to create dir [%s]: %v", mountDir, err)
		}

		err := saveVolumeData(targetPath, volDataFileName, tc.data)

		if !tc.shouldFail && err != nil {
			t.Errorf("unexpected failure: %v", err)
		}
		// did file get created
		dataDir := getTargetPath(testPodUID, specVolID, plug.host)
		file := filepath.Join(dataDir, volDataFileName)
		if _, err := os.Stat(file); err != nil {
			t.Errorf("failed to create data dir: %v", err)
		}

		// validate content
		data, err := os.ReadFile(file)
		if !tc.shouldFail && err != nil {
			t.Errorf("failed to read data file: %v", err)
		}

		jsonData := new(bytes.Buffer)
		if err := json.NewEncoder(jsonData).Encode(tc.data); err != nil {
			t.Errorf("failed to encode json: %v", err)
		}
		if string(data) != jsonData.String() {
			t.Errorf("expecting encoded data %v, got %v", string(data), jsonData)
		}
	}
}

func TestCreateCSIOperationContext(t *testing.T) {
	testCases := []struct {
		name     string
		spec     *volume.Spec
		migrated string
	}{
		{
			name:     "test volume spec nil",
			spec:     nil,
			migrated: "false",
		},
		{
			name: "test volume normal spec with migrated true",
			spec: &volume.Spec{
				Migrated: true,
			},
			migrated: "true",
		},
		{
			name: "test volume normal spec with migrated false",
			spec: &volume.Spec{
				Migrated: false,
			},
			migrated: "false",
		},
	}
	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		timeout := time.Minute
		ctx, _ := createCSIOperationContext(tc.spec, timeout)

		additionalInfoVal := ctx.Value(additionalInfoKey)
		if additionalInfoVal == nil {
			t.Error("Could not load additional info from context")
		}
		additionalInfoV, ok := additionalInfoVal.(additionalInfo)
		if !ok {
			t.Errorf("Additional info type assertion fail, additionalInfo object: %v", additionalInfoVal)
		}
		migrated := additionalInfoV.Migrated
		if migrated != tc.migrated {
			t.Errorf("Expect migrated value: %v, got: %v", tc.migrated, migrated)
		}
	}
}

// TestFindGlobalMountDataFromPodMount covers the reconstruction fallback of
// issue #101791. The fallback follows the mount reference the pod-local bind
// mount holds, so it can only ever land on the global mount that this very
// volume was staged at. An earlier revision matched on the directory name
// instead, which silently paired an inline ephemeral volume with an unrelated
// PersistentVolume that happened to share its short name.
func TestFindGlobalMountDataFromPodMount(t *testing.T) {
	const (
		driver    = "test-driver"
		podUID    = "pod-uid"
		device    = "/dev/sdb"
		volHandle = "handle-of-the-staged-pv"
	)

	// setup builds a plugin whose fake mount table is the one given, and
	// returns the pod-local dir for volName plus the global mount dir.
	setup := func(t *testing.T, volName string, mountPoints func(podMount, globalMount string) []mount.MountPoint) (*csiPlugin, string, string) {
		t.Helper()
		plug, tmpDir := newTestPlugin(t, nil)
		t.Cleanup(func() { _ = os.RemoveAll(tmpDir) })

		// GetMountRefs resolves symlinks before looking a path up in the mount
		// table, and on some platforms the temp dir sits behind one, so build
		// every path in this test from the resolved root.
		if resolved, err := filepath.EvalSymlinks(tmpDir); err == nil {
			tmpDir = resolved
		}

		podLocalDir := filepath.Join(tmpDir, "pods", podUID, "volumes", "kubernetes.io~csi", volName)
		if err := os.MkdirAll(filepath.Join(podLocalDir, "mount"), 0o755); err != nil {
			t.Fatalf("setup pod-local dir: %v", err)
		}

		globalDataDir := filepath.Join(tmpDir, "plugins", CSIPluginName, driver, "somehash")
		if err := os.MkdirAll(filepath.Join(globalDataDir, globalMountInGlobalPath), 0o755); err != nil {
			t.Fatalf("setup global dir: %v", err)
		}
		if err := saveVolumeData(globalDataDir, volDataFileName, map[string]string{
			volDataKey.specVolID:           volName,
			volDataKey.volHandle:           volHandle,
			volDataKey.driverName:          driver,
			volDataKey.volumeLifecycleMode: string(storagev1.VolumeLifecyclePersistent),
		}); err != nil {
			t.Fatalf("save global vol_data.json: %v", err)
		}

		fake, ok := plug.host.GetMounter().(*mount.FakeMounter)
		if !ok {
			t.Fatalf("expected a fake mounter, got %T", plug.host.GetMounter())
		}
		fake.MountPoints = mountPoints(filepath.Join(podLocalDir, "mount"), filepath.Join(globalDataDir, globalMountInGlobalPath))

		return plug, podLocalDir, globalDataDir
	}

	// bound is the healthy layout: SetUpAt bind mounted the global mount into
	// the pod directory, so both share a device.
	bound := func(podMount, globalMount string) []mount.MountPoint {
		return []mount.MountPoint{
			{Device: device, Path: globalMount},
			{Device: device, Path: podMount},
		}
	}

	t.Run("follows the bind mount to the global mount", func(t *testing.T) {
		plug, podLocalDir, globalDataDir := setup(t, "staged-pv", bound)

		dir, data, err := findGlobalMountDataFromPodMount(plug.host, podLocalDir)
		if err != nil {
			t.Fatalf("findGlobalMountDataFromPodMount: %v", err)
		}
		if dir != globalDataDir {
			t.Errorf("dir: got %q, want %q", dir, globalDataDir)
		}
		if data[volDataKey.volHandle] != volHandle {
			t.Errorf("volHandle: got %q, want %q", data[volDataKey.volHandle], volHandle)
		}
	})

	t.Run("an inline volume does not pair with a PV of the same name", func(t *testing.T) {
		// The staged PV is named "data". So is an inline ephemeral volume in
		// some pod, which is legal: one name is a PV name, the other an entry
		// in a pod spec. An inline volume never stages a global mount, so its
		// pod-local mount shares a device with nothing under the plugin dir.
		plug, podLocalDir, _ := setup(t, "data", func(podMount, globalMount string) []mount.MountPoint {
			return []mount.MountPoint{
				{Device: device, Path: globalMount},
				{Device: "/dev/sdc", Path: podMount},
			}
		})

		_, data, err := findGlobalMountDataFromPodMount(plug.host, podLocalDir)
		if err == nil {
			t.Fatalf("inline volume was paired with an unrelated PV, recovering volumeHandle %q", data[volDataKey.volHandle])
		}
	})

	t.Run("no mount reference is reported, not guessed", func(t *testing.T) {
		plug, podLocalDir, _ := setup(t, "staged-pv", func(podMount, globalMount string) []mount.MountPoint {
			return nil
		})

		if _, _, err := findGlobalMountDataFromPodMount(plug.host, podLocalDir); err == nil {
			t.Fatal("expected an error when the pod mount has no references, got none")
		}
	})

	t.Run("a mount reference with no volume data is skipped", func(t *testing.T) {
		plug, podLocalDir, globalDataDir := setup(t, "staged-pv", func(podMount, globalMount string) []mount.MountPoint {
			return []mount.MountPoint{
				{Device: device, Path: filepath.Join(t.TempDir(), globalMountInGlobalPath)},
				{Device: device, Path: globalMount},
				{Device: device, Path: podMount},
			}
		})

		dir, _, err := findGlobalMountDataFromPodMount(plug.host, podLocalDir)
		if err != nil {
			t.Fatalf("findGlobalMountDataFromPodMount: %v", err)
		}
		if dir != globalDataDir {
			t.Errorf("dir: got %q, want the real global mount %q", dir, globalDataDir)
		}
	})
}

// TestNewUnmounterFallsBackToGlobalMount is the other half of issue #101791.
// Reconstruction can rescue the volume, but the unmount that follows builds an
// unmounter from the very same pod-local vol_data.json, so without the same
// fallback the unmount operation is never generated, the pod is never dropped
// from the actual state of the world, and the global mount stays orphaned.
func TestNewUnmounterFallsBackToGlobalMount(t *testing.T) {
	const (
		driver    = "test-driver"
		podUID    = types.UID("pod-uid")
		specVolID = "staged-pv"
		volHandle = "handle-of-the-staged-pv"
		device    = "/dev/sdb"
	)

	setup := func(t *testing.T, gateOn bool) (*csiPlugin, string) {
		t.Helper()
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeReconstructionFallback, gateOn)
		registerFakePlugin(driver, "endpoint", []string{"1.0.0"}, t)

		plug, tmpDir := newTestPlugin(t, nil)
		t.Cleanup(func() { _ = os.RemoveAll(tmpDir) })
		if resolved, err := filepath.EvalSymlinks(tmpDir); err == nil {
			tmpDir = resolved
		}

		// Pod dir exists with its mount, but no vol_data.json.
		podMount := filepath.Join(tmpDir, "pods", string(podUID), "volumes", "kubernetes.io~csi", specVolID, "mount")
		if err := os.MkdirAll(podMount, 0o755); err != nil {
			t.Fatalf("setup pod-local dir: %v", err)
		}

		globalDataDir := filepath.Join(tmpDir, "plugins", CSIPluginName, driver, "somehash")
		if err := os.MkdirAll(filepath.Join(globalDataDir, globalMountInGlobalPath), 0o755); err != nil {
			t.Fatalf("setup global dir: %v", err)
		}
		if err := saveVolumeData(globalDataDir, volDataFileName, map[string]string{
			volDataKey.specVolID:           specVolID,
			volDataKey.volHandle:           volHandle,
			volDataKey.driverName:          driver,
			volDataKey.volumeLifecycleMode: string(storagev1.VolumeLifecyclePersistent),
		}); err != nil {
			t.Fatalf("save global vol_data.json: %v", err)
		}

		fake, ok := plug.host.GetMounter().(*mount.FakeMounter)
		if !ok {
			t.Fatalf("expected a fake mounter, got %T", plug.host.GetMounter())
		}
		fake.MountPoints = []mount.MountPoint{
			{Device: device, Path: filepath.Join(globalDataDir, globalMountInGlobalPath)},
			{Device: device, Path: podMount},
		}
		return plug, globalDataDir
	}

	t.Run("gate on, unmounter is built from the global mount", func(t *testing.T) {
		plug, _ := setup(t, true)

		unmounter, err := plug.NewUnmounter(specVolID, podUID)
		if err != nil {
			t.Fatalf("NewUnmounter: %v", err)
		}
		mgr, ok := unmounter.(*csiMountMgr)
		if !ok {
			t.Fatalf("expected a csiMountMgr, got %T", unmounter)
		}
		if string(mgr.driverName) != driver {
			t.Errorf("driverName: got %q, want %q", mgr.driverName, driver)
		}
		if mgr.volumeID != volHandle {
			t.Errorf("volumeID: got %q, want %q", mgr.volumeID, volHandle)
		}
	})

	t.Run("gate off, behaviour is unchanged", func(t *testing.T) {
		plug, _ := setup(t, false)

		if _, err := plug.NewUnmounter(specVolID, podUID); err == nil {
			t.Fatal("NewUnmounter succeeded with the gate off; the fallback must not run unless the feature is enabled")
		}
	})
}
