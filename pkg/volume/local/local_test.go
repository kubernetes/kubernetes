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
	"path"
	"syscall"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

const (
	testPVName    = "pvA"
	testMountPath = "pods/poduid/volumes/kubernetes.io~local-volume/pvA"
	testNodeName  = "fakeNodeName"
)

func getPlugin(t *testing.T) (string, volume.VolumePlugin) {
	tmpDir, err := utiltesting.MkTmpdir("localVolumeTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

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

func getPersistentPlugin(t *testing.T) (string, volume.PersistentVolumePlugin) {
	tmpDir, err := utiltesting.MkTmpdir("localVolumeTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

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

func getTestVolume(readOnly bool, path string) *volume.Spec {
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
		},
	}
	return volume.NewSpecFromPersistentVolume(pv, readOnly)
}

func contains(modes []v1.PersistentVolumeAccessMode, mode v1.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

func TestCanSupport(t *testing.T) {
	tmpDir, plug := getPlugin(t)
	defer os.RemoveAll(tmpDir)

	if !plug.CanSupport(getTestVolume(false, tmpDir)) {
		t.Errorf("Expected true")
	}
}

func TestGetAccessModes(t *testing.T) {
	tmpDir, plug := getPersistentPlugin(t)
	defer os.RemoveAll(tmpDir)

	modes := plug.GetAccessModes()
	if !contains(modes, v1.ReadWriteOnce) {
		t.Errorf("Expected AccessModeType %q", v1.ReadWriteOnce)
	}

	if contains(modes, v1.ReadWriteMany) {
		t.Errorf("Found AccessModeType %q, expected not", v1.ReadWriteMany)
	}
	if contains(modes, v1.ReadOnlyMany) {
		t.Errorf("Found AccessModeType %q, expected not", v1.ReadOnlyMany)
	}
}

func TestGetVolumeName(t *testing.T) {
	tmpDir, plug := getPersistentPlugin(t)
	defer os.RemoveAll(tmpDir)

	volName, err := plug.GetVolumeName(getTestVolume(false, tmpDir))
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
	mounter, err := plug.NewMounter(getTestVolume(false, "/no/backsteps/allowed/.."), pod, volume.VolumeOptions{})
	if err != nil {
		t.Fatal(err)
	}

	err = mounter.SetUp(nil)
	expectedMsg := "invalid path: /no/backsteps/allowed/.. must not contain '..'"
	if err.Error() != expectedMsg {
		t.Fatalf("expected error `%s` but got `%s`", expectedMsg, err)
	}
}

func TestMountUnmount(t *testing.T) {
	tmpDir, plug := getPlugin(t)
	defer os.RemoveAll(tmpDir)

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.NewMounter(getTestVolume(false, tmpDir), pod, volume.VolumeOptions{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Fatalf("Got a nil Mounter")
	}

	volPath := path.Join(tmpDir, testMountPath)
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
		t.Errorf("SetUp() failed: %v", err)
	}
}

func testFSGroupMount(plug volume.VolumePlugin, pod *v1.Pod, tmpDir string, fsGroup int64) error {
	mounter, err := plug.NewMounter(getTestVolume(false, tmpDir), pod, volume.VolumeOptions{})
	if err != nil {
		return err
	}
	if mounter == nil {
		return fmt.Errorf("Got a nil Mounter")
	}

	volPath := path.Join(tmpDir, testMountPath)
	path := mounter.GetPath()
	if path != volPath {
		return fmt.Errorf("Got unexpected path: %s", path)
	}

	if err := mounter.SetUp(&fsGroup); err != nil {
		return err
	}
	return nil
}

func TestFSGroupMount(t *testing.T) {
	tmpDir, plug := getPlugin(t)
	defer os.RemoveAll(tmpDir)
	info, err := os.Stat(tmpDir)
	if err != nil {
		t.Errorf("Error getting stats for %s (%v)", tmpDir, err)
	}
	s := info.Sys().(*syscall.Stat_t)
	if s == nil {
		t.Errorf("Error getting stats for %s (%v)", tmpDir, err)
	}
	fsGroup1 := int64(s.Gid)
	fsGroup2 := fsGroup1 + 1
	pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	pod1.Spec.SecurityContext = &v1.PodSecurityContext{
		FSGroup: &fsGroup1,
	}
	pod2 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	pod2.Spec.SecurityContext = &v1.PodSecurityContext{
		FSGroup: &fsGroup2,
	}
	err = testFSGroupMount(plug, pod1, tmpDir, fsGroup1)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	err = testFSGroupMount(plug, pod2, tmpDir, fsGroup2)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	//Checking if GID of tmpDir has not been changed by mounting it by second pod
	s = info.Sys().(*syscall.Stat_t)
	if s == nil {
		t.Errorf("Error getting stats for %s (%v)", tmpDir, err)
	}
	if fsGroup1 != int64(s.Gid) {
		t.Errorf("Old Gid %d for volume %s got overwritten by new Gid %d", fsGroup1, tmpDir, int64(s.Gid))
	}
}

func TestConstructVolumeSpec(t *testing.T) {
	tmpDir, plug := getPlugin(t)
	defer os.RemoveAll(tmpDir)

	volPath := path.Join(tmpDir, testMountPath)
	spec, err := plug.ConstructVolumeSpec(testPVName, volPath)
	if err != nil {
		t.Errorf("ConstructVolumeSpec() failed: %v", err)
	}
	if spec == nil {
		t.Fatalf("ConstructVolumeSpec() returned nil")
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

	ls := pv.Spec.PersistentVolumeSource.Local
	if ls == nil {
		t.Fatalf("LocalVolumeSource object nil")
	}
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	tmpDir, plug := getPlugin(t)
	defer os.RemoveAll(tmpDir)

	// Read only == true
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.NewMounter(getTestVolume(true, tmpDir), pod, volume.VolumeOptions{})
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
	mounter, err = plug.NewMounter(getTestVolume(false, tmpDir), pod, volume.VolumeOptions{})
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
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))
	spec := getTestVolume(false, tmpDir)

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

	createPlug, err := plugMgr.FindCreatablePluginBySpec(spec)
	if err == nil && createPlug != nil {
		t.Errorf("Creatable plugin found, expected none")
	}

	provisionPlug, err := plugMgr.FindProvisionablePluginByName(localVolumePluginName)
	if err == nil && provisionPlug != nil {
		t.Errorf("Provisionable plugin found, expected none")
	}
}
