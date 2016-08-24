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

package cache

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/snapshot/testing"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
)

// Calls AddVolume() once.
// Verifies a single volume entry exists.
func Test_AddVolume_Positive_NewVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)

	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)

	pvcName := "mypvc"
	pvcNamespace := "mynamespace"
	pvc := controllervolumetesting.GetTestPvc(pvcName, pvcNamespace, string(volumeName))

	snapshotName := "snapshot-name"

	// Act
	generatedVolumeName, err := asw.AddVolume(volumeSpec, pvc, snapshotName)

	// Assert
	if err != nil {
		t.Fatalf("AddVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	volumeExists := asw.VolumeExists(generatedVolumeName)
	if !volumeExists {
		t.Fatalf("%q volume does not exist, it should.", generatedVolumeName)
	}

	volumesToSnapshot := asw.GetVolumesToSnapshot()
	if len(volumesToSnapshot) != 1 {
		t.Fatalf("len(volumesToSnapshot) Expected: <1> Actual: <%v>", len(volumesToSnapshot))
	}
}

// Calls AddVolume() twice. Uses the same volume both times, but different snapshot names.
// Verifies a single volume entry exists.
func Test_AddVolume_Positive_ExistingVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)

	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)

	pvcName := "mypvc"
	pvcNamespace := "mynamespace"
	pvc := controllervolumetesting.GetTestPvc(pvcName, pvcNamespace, string(volumeName))

	snapshotName1 := "snapshot-name-1"
	snapshotName2 := "snapshot-name-2"

	// Act
	generatedVolumeName1, add1Err := asw.AddVolume(volumeSpec, pvc, snapshotName1)
	generatedVolumeName2, add2Err := asw.AddVolume(volumeSpec, pvc, snapshotName2)

	// Assert
	if add1Err != nil {
		t.Fatalf("AddVolume failed. Expected: <no error> Actual: <%v>", add1Err)
	}
	if add2Err != nil {
		t.Fatalf("AddVolume failed. Expected: <no error> Actual: <%v>", add2Err)
	}

	if generatedVolumeName1 != generatedVolumeName2 {
		t.Fatalf(
			"Generated volume names for the same volume should be the same but they are not: %q and %q",
			generatedVolumeName1,
			generatedVolumeName2)
	}

	volumeExists := asw.VolumeExists(generatedVolumeName1)
	if !volumeExists {
		t.Fatalf("%q volume does not exist, it should.", generatedVolumeName1)
	}

	volumesToSnapshot := asw.GetVolumesToSnapshot()
	if len(volumesToSnapshot) != 1 {
		t.Fatalf("len(volumesToSnapshot) Expected: <1> Actual: <%v>", len(volumesToSnapshot))
	}
}

// Populates data struct with one volume entry.
// Calls DeleteVolume() to delete volume.
// Verifies no volume entries exists.
func Test_DeleteVolume_Positive_VolumeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)

	pvcName := "mypvc"
	pvcNamespace := "mynamespace"
	pvc := controllervolumetesting.GetTestPvc(pvcName, pvcNamespace, string(volumeName))

	snapshotName := "snapshot-name"

	// Act
	generatedVolumeName, err := asw.AddVolume(volumeSpec, pvc, snapshotName)

	// Assert
	if err != nil {
		t.Fatalf("AddVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	volumeExists := asw.VolumeExists(generatedVolumeName)
	if !volumeExists {
		t.Fatalf("%q volume does not exist, it should.", generatedVolumeName)
	}

	// Act
	asw.DeleteVolume(generatedVolumeName)

	// Assert
	volumeStillExists := asw.VolumeExists(generatedVolumeName)
	if volumeStillExists {
		t.Fatalf("%q volume exists, it should not.", generatedVolumeName)
	}

	volumesToSnapshot := asw.GetVolumesToSnapshot()
	if len(volumesToSnapshot) != 0 {
		t.Fatalf("len(volumesToSnapshot) Expected: <1> Actual: <%v>", len(volumesToSnapshot))
	}
}

// Populates data struct with two volume entries (different persistent volume).
// Calls GetVolumesToSnapshot() to get list of entries.
// Verifies both volume entries are returned.
func Test_GetVolumesToSnapshot_Positive_TwoClaimsTwoVolumes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)

	volumeName1 := api.UniqueVolumeName("volume-name-1")
	volumeSpec1 := controllervolumetesting.GetTestVolumeSpec(string(volumeName1), volumeName1)
	pvcName1 := "mypvc1"
	pvcNamespace := "mynamespace"
	pvc1 := controllervolumetesting.GetTestPvc(pvcName1, pvcNamespace, string(volumeName1))
	snapshotName1 := "snapshot-name-1"
	_, err1 := asw.AddVolume(volumeSpec1, pvc1, snapshotName1)
	if err1 != nil {
		t.Fatalf("AddVolume1 failed. Expected: <no error> Actual: <%v>", err1)
	}

	volumeName2 := api.UniqueVolumeName("volume-name-2")
	volumeSpec2 := controllervolumetesting.GetTestVolumeSpec(string(volumeName2), volumeName2)
	pvcName2 := "mypvc2"
	pvc2 := controllervolumetesting.GetTestPvc(pvcName2, pvcNamespace, string(volumeName2))
	snapshotName2 := "snapshot-name-2"
	_, err2 := asw.AddVolume(volumeSpec2, pvc2, snapshotName2)
	if err2 != nil {
		t.Fatalf("AddVolume2 failed. Expected: <no error> Actual: <%v>", err2)
	}

	// Act
	volumesToSnapshot := asw.GetVolumesToSnapshot()

	// Assert
	if len(volumesToSnapshot) != 2 {
		t.Fatalf("len(volumesToSnapshot) Expected: <2> Actual: <%v>", len(volumesToSnapshot))
	}
}

// Populates data struct with two volume entries (same persistent volume).
// Calls GetVolumesToSnapshot() to get list of entries.
// Verifies only one volume entry is returned.
func Test_GetVolumesToSnapshot_Positive_TwoClaimsOneVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)

	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)

	pvcName1 := "mypvc1"
	pvcNamespace := "mynamespace"
	pvc1 := controllervolumetesting.GetTestPvc(pvcName1, pvcNamespace, string(volumeName))
	snapshotName1 := "snapshot-name-1"
	_, err1 := asw.AddVolume(volumeSpec, pvc1, snapshotName1)
	if err1 != nil {
		t.Fatalf("AddVolume1 failed. Expected: <no error> Actual: <%v>", err1)
	}

	pvcName2 := "mypvc2"
	pvc2 := controllervolumetesting.GetTestPvc(pvcName2, pvcNamespace, string(volumeName))
	snapshotName2 := "snapshot-name-2"
	_, err2 := asw.AddVolume(volumeSpec, pvc2, snapshotName2)
	if err2 != nil {
		t.Fatalf("AddVolume2 failed. Expected: <no error> Actual: <%v>", err2)
	}

	// Act
	volumesToSnapshot := asw.GetVolumesToSnapshot()

	// Assert
	if len(volumesToSnapshot) != 1 {
		t.Fatalf("len(volumesToSnapshot) Expected: <1> Actual: <%v>", len(volumesToSnapshot))
	}
}
