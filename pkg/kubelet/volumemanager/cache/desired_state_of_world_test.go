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

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

// Calls AddPodToVolume() to add new pod to new volume
// Verifies newly added pod/volume exists via
// PodExistsInVolume() VolumeExists() and GetVolumesToMount()
func Test_AddPodToVolume_Positive_NewPodNewVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	pod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: "pod3",
			UID:  "pod3uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	podName := volumehelper.GetUniquePodName(pod)

	// Act
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsDsw(t, generatedVolumeName, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolumeName, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, podName, generatedVolumeName, dsw)
}

// Calls AddPodToVolume() twice to add the same pod to the same volume
// Verifies newly added pod/volume exists via
// PodExistsInVolume() VolumeExists() and GetVolumesToMount() and no errors.
func Test_AddPodToVolume_Positive_ExistingPodExistingVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	pod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: "pod3",
			UID:  "pod3uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	podName := volumehelper.GetUniquePodName(pod)

	// Act
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsDsw(t, generatedVolumeName, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolumeName, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, podName, generatedVolumeName, dsw)
}

// Populates data struct with a new volume/pod
// Calls DeletePodFromVolume() to removes the pod
// Verifies newly added pod/volume are deleted
func Test_DeletePodFromVolume_Positive_PodExistsVolumeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	pod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: "pod3",
			UID:  "pod3uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	podName := volumehelper.GetUniquePodName(pod)
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */)
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}
	verifyVolumeExistsDsw(t, generatedVolumeName, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolumeName, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, podName, generatedVolumeName, dsw)

	// Act
	dsw.DeletePodFromVolume(podName, generatedVolumeName)

	// Assert
	verifyVolumeDoesntExist(t, generatedVolumeName, dsw)
	verifyVolumeDoesntExistInVolumesToMount(t, generatedVolumeName, dsw)
	verifyPodDoesntExistInVolumeDsw(t, podName, generatedVolumeName, dsw)
}

// Calls AddPodToVolume() to add three new volumes to data struct
// Verifies newly added pod/volume exists via PodExistsInVolume()
// VolumeExists() and GetVolumesToMount()
// Marks only second volume as reported in use.
// Verifies only that volume is marked reported in use
// Marks only first volume as reported in use.
// Verifies only that volume is marked reported in use
func Test_MarkVolumesReportedInUse_Positive_NewPodNewVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)

	pod1 := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume1-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volume1Spec := &volume.Spec{Volume: &pod1.Spec.Volumes[0]}
	pod1Name := volumehelper.GetUniquePodName(pod1)

	pod2 := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: "pod2",
			UID:  "pod2uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume2-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device2",
						},
					},
				},
			},
		},
	}

	volume2Spec := &volume.Spec{Volume: &pod2.Spec.Volumes[0]}
	pod2Name := volumehelper.GetUniquePodName(pod2)

	pod3 := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: "pod3",
			UID:  "pod3uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume3-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device3",
						},
					},
				},
			},
		},
	}

	volume3Spec := &volume.Spec{Volume: &pod3.Spec.Volumes[0]}
	pod3Name := volumehelper.GetUniquePodName(pod3)

	generatedVolume1Name, err := dsw.AddPodToVolume(
		pod1Name, pod1, volume1Spec, volume1Spec.Name(), "" /* volumeGidValue */)
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	generatedVolume2Name, err := dsw.AddPodToVolume(
		pod2Name, pod2, volume2Spec, volume2Spec.Name(), "" /* volumeGidValue */)
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	generatedVolume3Name, err := dsw.AddPodToVolume(
		pod3Name, pod3, volume3Spec, volume3Spec.Name(), "" /* volumeGidValue */)
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	volumesReportedInUse := []v1.UniqueVolumeName{generatedVolume2Name}
	dsw.MarkVolumesReportedInUse(volumesReportedInUse)

	// Assert
	verifyVolumeExistsDsw(t, generatedVolume1Name, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolume1Name, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, pod1Name, generatedVolume1Name, dsw)
	verifyVolumeExistsDsw(t, generatedVolume2Name, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolume2Name, true /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, pod2Name, generatedVolume2Name, dsw)
	verifyVolumeExistsDsw(t, generatedVolume3Name, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolume3Name, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, pod3Name, generatedVolume3Name, dsw)

	// Act
	volumesReportedInUse = []v1.UniqueVolumeName{generatedVolume3Name}
	dsw.MarkVolumesReportedInUse(volumesReportedInUse)

	// Assert
	verifyVolumeExistsDsw(t, generatedVolume1Name, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolume1Name, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, pod1Name, generatedVolume1Name, dsw)
	verifyVolumeExistsDsw(t, generatedVolume2Name, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolume2Name, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, pod2Name, generatedVolume2Name, dsw)
	verifyVolumeExistsDsw(t, generatedVolume3Name, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolume3Name, true /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, pod3Name, generatedVolume3Name, dsw)
}

func verifyVolumeExistsDsw(
	t *testing.T, expectedVolumeName v1.UniqueVolumeName, dsw DesiredStateOfWorld) {
	volumeExists := dsw.VolumeExists(expectedVolumeName)
	if !volumeExists {
		t.Fatalf(
			"VolumeExists(%q) failed. Expected: <true> Actual: <%v>",
			expectedVolumeName,
			volumeExists)
	}
}

func verifyVolumeDoesntExist(
	t *testing.T, expectedVolumeName v1.UniqueVolumeName, dsw DesiredStateOfWorld) {
	volumeExists := dsw.VolumeExists(expectedVolumeName)
	if volumeExists {
		t.Fatalf(
			"VolumeExists(%q) returned incorrect value. Expected: <false> Actual: <%v>",
			expectedVolumeName,
			volumeExists)
	}
}

func verifyVolumeExistsInVolumesToMount(
	t *testing.T,
	expectedVolumeName v1.UniqueVolumeName,
	expectReportedInUse bool,
	dsw DesiredStateOfWorld) {
	volumesToMount := dsw.GetVolumesToMount()
	for _, volume := range volumesToMount {
		if volume.VolumeName == expectedVolumeName {
			if volume.ReportedInUse != expectReportedInUse {
				t.Fatalf(
					"Found volume %v in the list of VolumesToMount, but ReportedInUse incorrect. Expected: <%v> Actual: <%v>",
					expectedVolumeName,
					expectReportedInUse,
					volume.ReportedInUse)
			}

			return
		}
	}

	t.Fatalf(
		"Could not find volume %v in the list of desired state of world volumes to mount %+v",
		expectedVolumeName,
		volumesToMount)
}

func verifyVolumeDoesntExistInVolumesToMount(
	t *testing.T, volumeToCheck v1.UniqueVolumeName, dsw DesiredStateOfWorld) {
	volumesToMount := dsw.GetVolumesToMount()
	for _, volume := range volumesToMount {
		if volume.VolumeName == volumeToCheck {
			t.Fatalf(
				"Found volume %v in the list of desired state of world volumes to mount. Expected it not to exist.",
				volumeToCheck)
		}
	}
}

func verifyPodExistsInVolumeDsw(
	t *testing.T,
	expectedPodName volumetypes.UniquePodName,
	expectedVolumeName v1.UniqueVolumeName,
	dsw DesiredStateOfWorld) {
	if podExistsInVolume := dsw.PodExistsInVolume(
		expectedPodName, expectedVolumeName); !podExistsInVolume {
		t.Fatalf(
			"DSW PodExistsInVolume returned incorrect value. Expected: <true> Actual: <%v>",
			podExistsInVolume)
	}
}

func verifyPodDoesntExistInVolumeDsw(
	t *testing.T,
	expectedPodName volumetypes.UniquePodName,
	expectedVolumeName v1.UniqueVolumeName,
	dsw DesiredStateOfWorld) {
	if podExistsInVolume := dsw.PodExistsInVolume(
		expectedPodName, expectedVolumeName); podExistsInVolume {
		t.Fatalf(
			"DSW PodExistsInVolume returned incorrect value. Expected: <true> Actual: <%v>",
			podExistsInVolume)
	}
}
