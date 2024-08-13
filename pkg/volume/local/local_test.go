//go:build linux || darwin || windows
// +build linux darwin windows

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

package local

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/hostutil"
	"k8s.io/mount-utils"
)

const (
	testPVName                        = "pvA"
	testMountPath                     = "pods/poduid/volumes/kubernetes.io~local-volume/pvA"
	testGlobalPath                    = "plugins/kubernetes.io~local-volume/volumeDevices/pvA"
	testPodPath                       = "pods/poduid/volumeDevices/kubernetes.io~local-volume"
	testBlockFormattingToFSGlobalPath = "plugins/kubernetes.io/local-volume/mounts/pvA"
)

func getPlugin(t *testing.T) (string, volume.VolumePlugin) {
	tmpDir, err := utiltesting.MkTmpdir("localVolumeTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeKubeletVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName(localVolumePluginName)
	if err != nil {
		os.RemoveAll(tmpDir)
		t.Fatalf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != localVolumePluginName {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	return tmpDir, plug
}

func getBlockPlugin(t *testing.T) (string, volume.BlockVolumePlugin) {
	tmpDir, err := utiltesting.MkTmpdir("localVolumeTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeKubeletVolumeHost(t, tmpDir, nil, nil))
	plug, err := plugMgr.FindMapperPluginByName(localVolumePluginName)
	if err != nil {
		os.RemoveAll(tmpDir)
		t.Fatalf("Can't find the plugin by name: %q", localVolumePluginName)
	}
	if plug.GetPluginName() != localVolumePluginName {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	return tmpDir, plug
}

func getNodeExpandablePlugin(t *testing.T, isBlockDevice bool) (string, volume.NodeExpandableVolumePlugin) {
	tmpDir, err := utiltesting.MkTmpdir("localVolumeTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}

	plugMgr := volume.VolumePluginMgr{}
	var pathToFSType map[string]hostutil.FileType
	if isBlockDevice {
		pathToFSType = map[string]hostutil.FileType{
			tmpDir: hostutil.FileTypeBlockDev,
		}
	} else {
		pathToFSType = map[string]hostutil.FileType{
			tmpDir: hostutil.FileTypeDirectory,
		}
	}

	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeKubeletVolumeHostWithMounterFSType(t, tmpDir, nil, nil, pathToFSType))

	plug, err := plugMgr.FindNodeExpandablePluginByName(localVolumePluginName)
	if err != nil {
		os.RemoveAll(tmpDir)
		t.Fatalf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != localVolumePluginName {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	return tmpDir, plug
}

func getPersistentPlugin(t *testing.T) (string, volume.PersistentVolumePlugin) {
	tmpDir, err := utiltesting.MkTmpdir("localVolumeTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeKubeletVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName(localVolumePluginName)
	if err != nil {
		os.RemoveAll(tmpDir)
		t.Fatalf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != localVolumePluginName {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	return tmpDir, plug
}

func getDeviceMountablePluginWithBlockPath(t *testing.T, isBlockDevice bool) (string, volume.DeviceMountableVolumePlugin) {
	var (
		source string
		err    error
	)

	if isBlockDevice && runtime.GOOS == "windows" {
		// On Windows, block devices are referenced by the disk number, which is validated by the mounter,
		source = "0"
	} else {
		source, err = utiltesting.MkTmpdir("localVolumeTest")
		if err != nil {
			t.Fatalf("can't make a temp dir: %v", err)
		}
	}

	plugMgr := volume.VolumePluginMgr{}
	var pathToFSType map[string]hostutil.FileType
	if isBlockDevice {
		pathToFSType = map[string]hostutil.FileType{
			source: hostutil.FileTypeBlockDev,
		}
	}

	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeKubeletVolumeHostWithMounterFSType(t, source, nil, nil, pathToFSType))

	plug, err := plugMgr.FindDeviceMountablePluginByName(localVolumePluginName)
	if err != nil {
		os.RemoveAll(source)
		t.Fatalf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != localVolumePluginName {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	return source, plug
}

func getTestVolume(readOnly bool, path string, isBlock bool, mountOptions []string) *volume.Spec {
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: testPVName,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				Local: &v1.LocalVolumeSource{
					Path: path,
				},
			},
			MountOptions: mountOptions,
		},
	}

	if isBlock {
		blockMode := v1.PersistentVolumeBlock
		pv.Spec.VolumeMode = &blockMode
	} else {
		fsMode := v1.PersistentVolumeFilesystem
		pv.Spec.VolumeMode = &fsMode
	}
	return volume.NewSpecFromPersistentVolume(pv, readOnly)
}

func TestCanSupport(t *testing.T) {
	tmpDir, plug := getPlugin(t)
	defer os.RemoveAll(tmpDir)

	if !plug.CanSupport(getTestVolume(false, tmpDir, false, nil)) {
		t.Errorf("Expected true")
	}
}

func TestGetAccessModes(t *testing.T) {
	tmpDir, plug := getPersistentPlugin(t)
	defer os.RemoveAll(tmpDir)

	modes := plug.GetAccessModes()
	if !volumetest.ContainsAccessMode(modes, v1.ReadWriteOnce) {
		t.Errorf("Expected AccessModeType %q", v1.ReadWriteOnce)
	}

	if volumetest.ContainsAccessMode(modes, v1.ReadWriteMany) {
		t.Errorf("Found AccessModeType %q, expected not", v1.ReadWriteMany)
	}
	if volumetest.ContainsAccessMode(modes, v1.ReadOnlyMany) {
		t.Errorf("Found AccessModeType %q, expected not", v1.ReadOnlyMany)
	}
}

func TestGetVolumeName(t *testing.T) {
	tmpDir, plug := getPersistentPlugin(t)
	defer os.RemoveAll(tmpDir)

	volName, err := plug.GetVolumeName(getTestVolume(false, tmpDir, false, nil))
	if err != nil {
		t.Errorf("Failed to get volume name: %v", err)
	}
	if volName != testPVName {
		t.Errorf("Expected volume name %q, got %q", testPVName, volName)
	}
}

func TestInvalidLocalPath(t *testing.T) {
	tmpDir, plug := getPlugin(t)
	defer os.RemoveAll(tmpDir)

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.NewMounter(getTestVolume(false, "/no/backsteps/allowed/..", false, nil), pod)
	if err != nil {
		t.Fatal(err)
	}

	err = mounter.SetUp(volume.MounterArgs{})
	expectedMsg := "invalid path: /no/backsteps/allowed/.. must not contain '..'"
	if err.Error() != expectedMsg {
		t.Fatalf("expected error `%s` but got `%s`", expectedMsg, err)
	}
}

func TestBlockDeviceGlobalPathAndMountDevice(t *testing.T) {
	// Block device global mount path testing
	tmpBlockDir, plug := getDeviceMountablePluginWithBlockPath(t, true)
	defer os.RemoveAll(tmpBlockDir)

	dm, err := plug.NewDeviceMounter()
	if err != nil {
		t.Errorf("Failed to make a new device mounter: %v", err)
	}

	pvSpec := getTestVolume(false, tmpBlockDir, false, nil)

	expectedGlobalPath := filepath.Join(tmpBlockDir, testBlockFormattingToFSGlobalPath)
	actualPath, err := dm.GetDeviceMountPath(pvSpec)
	if err != nil {
		t.Errorf("Failed to get device mount path: %v", err)
	}
	if expectedGlobalPath != actualPath {
		t.Fatalf("Expected device mount global path:%s, got: %s", expectedGlobalPath, actualPath)
	}

	fmt.Println("expected global path is:", expectedGlobalPath)

	err = dm.MountDevice(pvSpec, tmpBlockDir, expectedGlobalPath, volume.DeviceMounterArgs{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(actualPath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("DeviceMounter.MountDevice() failed, device mount path not created: %s", actualPath)
		} else {
			t.Errorf("DeviceMounter.MountDevice() failed: %v", err)
		}
	}

	du, err := plug.NewDeviceUnmounter()
	if err != nil {
		t.Fatalf("Create device unmounter error: %v", err)
	}

	err = du.UnmountDevice(actualPath)
	if err != nil {
		t.Fatalf("Unmount device error: %v", err)
	}
}

func TestFSGlobalPathAndMountDevice(t *testing.T) {
	// FS global path testing
	tmpFSDir, plug := getDeviceMountablePluginWithBlockPath(t, false)
	defer os.RemoveAll(tmpFSDir)

	dm, err := plug.NewDeviceMounter()
	if err != nil {
		t.Errorf("Failed to make a new device mounter: %v", err)
	}

	pvSpec := getTestVolume(false, tmpFSDir, false, nil)

	expectedGlobalPath := tmpFSDir
	actualPath, err := dm.GetDeviceMountPath(pvSpec)
	if err != nil {
		t.Errorf("Failed to get device mount path: %v", err)
	}
	if expectedGlobalPath != actualPath {
		t.Fatalf("Expected device mount global path:%s, got: %s", expectedGlobalPath, actualPath)
	}

	// Actually, we will do nothing if the local path is FS type
	err = dm.MountDevice(pvSpec, tmpFSDir, expectedGlobalPath, volume.DeviceMounterArgs{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(expectedGlobalPath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("DeviceMounter.MountDevice() failed, device mount path not created: %s", expectedGlobalPath)
		} else {
			t.Errorf("DeviceMounter.MountDevice() failed: %v", err)
		}
	}
}

func TestNodeExpand(t *testing.T) {
	// FS global path testing
	tmpFSDir, plug := getNodeExpandablePlugin(t, false)
	defer os.RemoveAll(tmpFSDir)

	pvSpec := getTestVolume(false, tmpFSDir, false, nil)

	resizeOptions := volume.NodeResizeOptions{
		VolumeSpec: pvSpec,
		DevicePath: tmpFSDir,
	}

	// Actually, we will do no volume expansion if volume is of type dir
	resizeDone, err := plug.NodeExpand(resizeOptions)
	if err != nil {
		t.Fatal(err)
	}
	if !resizeDone {
		t.Errorf("expected resize to be done")
	}
}

func TestMountUnmount(t *testing.T) {
	tmpDir, plug := getPlugin(t)
	defer os.RemoveAll(tmpDir)

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.NewMounter(getTestVolume(false, tmpDir, false, nil), pod)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Fatalf("Got a nil Mounter")
	}

	volPath := filepath.Join(tmpDir, testMountPath)
	path := mounter.GetPath()
	if path != volPath {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err := mounter.SetUp(volume.MounterArgs{}); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}

	if runtime.GOOS != "windows" {
		// skip this check in windows since the "bind mount" logic is implemented differently in mount_windows.go
		if _, err := os.Stat(path); err != nil {
			if os.IsNotExist(err) {
				t.Errorf("SetUp() failed, volume path not created: %s", path)
			} else {
				t.Errorf("SetUp() failed: %v", err)
			}
		}
	}

	unmounter, err := plug.NewUnmounter(testPVName, pod.UID)
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Fatalf("Got a nil Unmounter")
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

// TestMapUnmap tests block map and unmap interfaces.
func TestMapUnmap(t *testing.T) {
	tmpDir, plug := getBlockPlugin(t)
	defer os.RemoveAll(tmpDir)

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	volSpec := getTestVolume(false, tmpDir, true /*isBlock*/, nil)
	mapper, err := plug.NewBlockVolumeMapper(volSpec, pod)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mapper == nil {
		t.Fatalf("Got a nil Mounter")
	}

	expectedGlobalPath := filepath.Join(tmpDir, testGlobalPath)
	globalPath, err := mapper.GetGlobalMapPath(volSpec)
	if err != nil {
		t.Errorf("Failed to get global path: %v", err)
	}
	if globalPath != expectedGlobalPath {
		t.Errorf("Got unexpected path: %s, expected %s", globalPath, expectedGlobalPath)
	}
	expectedPodPath := filepath.Join(tmpDir, testPodPath)
	podPath, volName := mapper.GetPodDeviceMapPath()
	if podPath != expectedPodPath {
		t.Errorf("Got unexpected pod path: %s, expected %s", podPath, expectedPodPath)
	}
	if volName != testPVName {
		t.Errorf("Got unexpected volNamne: %s, expected %s", volName, testPVName)
	}
	var devPath string

	if customMapper, ok := mapper.(volume.CustomBlockVolumeMapper); ok {
		_, err = customMapper.SetUpDevice()
		if err != nil {
			t.Errorf("Failed to SetUpDevice, err: %v", err)
		}
		devPath, err = customMapper.MapPodDevice()
		if err != nil {
			t.Errorf("Failed to MapPodDevice, err: %v", err)
		}
	}

	if _, err := os.Stat(devPath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUpDevice() failed, volume path not created: %s", devPath)
		} else {
			t.Errorf("SetUpDevice() failed: %v", err)
		}
	}

	unmapper, err := plug.NewBlockVolumeUnmapper(testPVName, pod.UID)
	if err != nil {
		t.Fatalf("Failed to make a new Unmapper: %v", err)
	}
	if unmapper == nil {
		t.Fatalf("Got a nil Unmapper")
	}

	if customUnmapper, ok := unmapper.(volume.CustomBlockVolumeUnmapper); ok {
		if err := customUnmapper.UnmapPodDevice(); err != nil {
			t.Errorf("UnmapPodDevice failed, err: %v", err)
		}

		if err := customUnmapper.TearDownDevice(globalPath, devPath); err != nil {
			t.Errorf("TearDownDevice failed, err: %v", err)
		}
	}
}

func testFSGroupMount(plug volume.VolumePlugin, pod *v1.Pod, tmpDir string, fsGroup int64) error {
	mounter, err := plug.NewMounter(getTestVolume(false, tmpDir, false, nil), pod)
	if err != nil {
		return err
	}
	if mounter == nil {
		return fmt.Errorf("got a nil Mounter")
	}

	volPath := filepath.Join(tmpDir, testMountPath)
	path := mounter.GetPath()
	if path != volPath {
		return fmt.Errorf("got unexpected path: %s", path)
	}

	var mounterArgs volume.MounterArgs
	mounterArgs.FsGroup = &fsGroup
	if err := mounter.SetUp(mounterArgs); err != nil {
		return err
	}
	return nil
}

func TestConstructVolumeSpec(t *testing.T) {
	tests := []struct {
		name         string
		mountPoints  []mount.MountPoint
		expectedPath string
	}{
		{
			name: "filesystem volume with directory source",
			mountPoints: []mount.MountPoint{
				{
					Device: "/mnt/disk/ssd0",
					Path:   "pods/poduid/volumes/kubernetes.io~local-volume/pvA",
				},
			},
			expectedPath: "",
		},
		{
			name: "filesystem volume with block source",
			mountPoints: []mount.MountPoint{
				{
					Device: "/dev/loop0",
					Path:   testMountPath,
				},
				{
					Device: "/dev/loop0",
					Path:   testBlockFormattingToFSGlobalPath,
				},
			},
			expectedPath: "/dev/loop0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir, err := utiltesting.MkTmpdir("localVolumeTest")
			if err != nil {
				t.Fatalf("can't make a temp dir: %v", err)
			}
			defer os.RemoveAll(tmpDir)
			plug := &localVolumePlugin{
				host: volumetest.NewFakeKubeletVolumeHost(t, tmpDir, nil, nil),
			}
			mounter := plug.host.GetMounter(plug.GetPluginName())
			fakeMountPoints := []mount.MountPoint{}
			for _, mp := range tt.mountPoints {
				fakeMountPoint := mp
				fakeMountPoint.Path = filepath.Join(tmpDir, mp.Path)
				fakeMountPoints = append(fakeMountPoints, fakeMountPoint)
			}
			mounter.(*mount.FakeMounter).MountPoints = fakeMountPoints
			volPath := filepath.Join(tmpDir, testMountPath)
			rec, err := plug.ConstructVolumeSpec(testPVName, volPath)
			if err != nil {
				t.Errorf("ConstructVolumeSpec() failed: %v", err)
			}
			if rec.Spec == nil {
				t.Fatalf("ConstructVolumeSpec() returned nil")
			}

			volName := rec.Spec.Name()
			if volName != testPVName {
				t.Errorf("Expected volume name %q, got %q", testPVName, volName)
			}

			if rec.Spec.Volume != nil {
				t.Errorf("Volume object returned, expected nil")
			}

			pv := rec.Spec.PersistentVolume
			if pv == nil {
				t.Fatalf("PersistentVolume object nil")
			}

			if rec.Spec.PersistentVolume.Spec.VolumeMode == nil {
				t.Fatalf("Volume mode has not been set.")
			}

			if *rec.Spec.PersistentVolume.Spec.VolumeMode != v1.PersistentVolumeFilesystem {
				t.Errorf("Unexpected volume mode %q", *rec.Spec.PersistentVolume.Spec.VolumeMode)
			}

			ls := pv.Spec.PersistentVolumeSource.Local
			if ls == nil {
				t.Fatalf("LocalVolumeSource object nil")
			}

			if pv.Spec.PersistentVolumeSource.Local.Path != tt.expectedPath {
				t.Fatalf("Unexpected path got %q, expected %q", pv.Spec.PersistentVolumeSource.Local.Path, tt.expectedPath)
			}
		})
	}

}

func TestConstructBlockVolumeSpec(t *testing.T) {
	tmpDir, plug := getBlockPlugin(t)
	defer os.RemoveAll(tmpDir)

	podPath := filepath.Join(tmpDir, testPodPath)
	spec, err := plug.ConstructBlockVolumeSpec(types.UID("poduid"), testPVName, podPath)
	if err != nil {
		t.Errorf("ConstructBlockVolumeSpec() failed: %v", err)
	}
	if spec == nil {
		t.Fatalf("ConstructBlockVolumeSpec() returned nil")
	}

	volName := spec.Name()
	if volName != testPVName {
		t.Errorf("Expected volume name %q, got %q", testPVName, volName)
	}

	if spec.Volume != nil {
		t.Errorf("Volume object returned, expected nil")
	}

	pv := spec.PersistentVolume
	if pv == nil {
		t.Fatalf("PersistentVolume object nil")
	}

	if spec.PersistentVolume.Spec.VolumeMode == nil {
		t.Fatalf("Volume mode has not been set.")
	}

	if *spec.PersistentVolume.Spec.VolumeMode != v1.PersistentVolumeBlock {
		t.Errorf("Unexpected volume mode %q", *spec.PersistentVolume.Spec.VolumeMode)
	}

	ls := pv.Spec.PersistentVolumeSource.Local
	if ls == nil {
		t.Fatalf("LocalVolumeSource object nil")
	}
}

func TestMountOptions(t *testing.T) {
	tmpDir, plug := getPlugin(t)
	defer os.RemoveAll(tmpDir)

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.NewMounter(getTestVolume(false, tmpDir, false, []string{"test-option"}), pod)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Fatalf("Got a nil Mounter")
	}

	// Wrap with FakeMounter.
	fakeMounter := mount.NewFakeMounter(nil)
	mounter.(*localVolumeMounter).mounter = fakeMounter

	if err := mounter.SetUp(volume.MounterArgs{}); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	mountOptions := fakeMounter.MountPoints[0].Opts
	expectedMountOptions := []string{"bind", "test-option"}
	if !reflect.DeepEqual(mountOptions, expectedMountOptions) {
		t.Errorf("Expected mount options to be %v got %v", expectedMountOptions, mountOptions)
	}
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	tmpDir, plug := getPlugin(t)
	defer os.RemoveAll(tmpDir)

	// Read only == true
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.NewMounter(getTestVolume(true, tmpDir, false, nil), pod)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Fatalf("Got a nil Mounter")
	}
	if !mounter.GetAttributes().ReadOnly {
		t.Errorf("Expected true for mounter.IsReadOnly")
	}

	// Read only == false
	mounter, err = plug.NewMounter(getTestVolume(false, tmpDir, false, nil), pod)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Fatalf("Got a nil Mounter")
	}
	if mounter.GetAttributes().ReadOnly {
		t.Errorf("Expected false for mounter.IsReadOnly")
	}
}

func TestUnsupportedPlugins(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("localVolumeTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeKubeletVolumeHost(t, tmpDir, nil, nil))
	spec := getTestVolume(false, tmpDir, false, nil)

	recyclePlug, err := plugMgr.FindRecyclablePluginBySpec(spec)
	if err == nil && recyclePlug != nil {
		t.Errorf("Recyclable plugin found, expected none")
	}

	deletePlug, err := plugMgr.FindDeletablePluginByName(localVolumePluginName)
	if err == nil && deletePlug != nil {
		t.Errorf("Deletable plugin found, expected none")
	}

	attachPlug, err := plugMgr.FindAttachablePluginByName(localVolumePluginName)
	if err == nil && attachPlug != nil {
		t.Errorf("Attachable plugin found, expected none")
	}

	provisionPlug, err := plugMgr.FindProvisionablePluginByName(localVolumePluginName)
	if err == nil && provisionPlug != nil {
		t.Errorf("Provisionable plugin found, expected none")
	}
}

func TestFilterPodMounts(t *testing.T) {
	tmpDir, plug := getPlugin(t)
	defer os.RemoveAll(tmpDir)

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.NewMounter(getTestVolume(false, tmpDir, false, nil), pod)
	if err != nil {
		t.Fatal(err)
	}
	lvMounter, ok := mounter.(*localVolumeMounter)
	if !ok {
		t.Fatal("mounter is not localVolumeMounter")
	}

	host := volumetest.NewFakeKubeletVolumeHost(t, tmpDir, nil, nil)
	podsDir := host.GetPodsDir()

	cases := map[string]struct {
		input    []string
		expected []string
	}{
		"empty": {
			[]string{},
			[]string{},
		},
		"not-pod-mount": {
			[]string{"/mnt/outside"},
			[]string{},
		},
		"pod-mount": {
			[]string{filepath.Join(podsDir, "pod-mount")},
			[]string{filepath.Join(podsDir, "pod-mount")},
		},
		"not-directory-prefix": {
			[]string{podsDir + "pod-mount"},
			[]string{},
		},
		"mix": {
			[]string{"/mnt/outside",
				filepath.Join(podsDir, "pod-mount"),
				"/another/outside",
				filepath.Join(podsDir, "pod-mount2")},
			[]string{filepath.Join(podsDir, "pod-mount"),
				filepath.Join(podsDir, "pod-mount2")},
		},
	}
	for name, test := range cases {
		output := lvMounter.filterPodMounts(test.input)
		if !reflect.DeepEqual(output, test.expected) {
			t.Errorf("%v failed: output %+v doesn't equal expected %+v", name, output, test.expected)
		}
	}
}
