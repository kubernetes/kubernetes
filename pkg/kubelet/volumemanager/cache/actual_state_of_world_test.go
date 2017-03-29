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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

var emptyVolumeName = v1.UniqueVolumeName("")

// Calls MarkVolumeAsAttached() once to add volume
// Verifies newly added volume exists in GetUnmountedVolumes()
// Verifies newly added volume doesn't exist in GetGloballyMountedVolumes()
func Test_MarkVolumeAsAttached_Positive_NewVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, plugin := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld("mynode" /* nodeName */, volumePluginMgr)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
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
	devicePath := "fake/device/path"
	generatedVolumeName, _ := volumehelper.GetUniqueVolumeNameFromSpec(plugin, volumeSpec)

	// Act
	err := asw.MarkVolumeAsAttached(emptyVolumeName, volumeSpec, "" /* nodeName */, devicePath)

	// Assert
	if err != nil {
		t.Fatalf("MarkVolumeAsAttached failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsAsw(t, generatedVolumeName, true /* shouldExist */, asw)
	verifyVolumeExistsInUnmountedVolumes(t, generatedVolumeName, asw)
	verifyVolumeDoesntExistInGloballyMountedVolumes(t, generatedVolumeName, asw)
}

// Calls MarkVolumeAsAttached() once to add volume, specifying a name --
// establishes that the supplied volume name is used to register the volume
// rather than the generated one.
// Verifies newly added volume exists in GetUnmountedVolumes()
// Verifies newly added volume doesn't exist in GetGloballyMountedVolumes()
func Test_MarkVolumeAsAttached_SuppliedVolumeName_Positive_NewVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld("mynode" /* nodeName */, volumePluginMgr)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
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
	devicePath := "fake/device/path"
	volumeName := v1.UniqueVolumeName("this-would-never-be-a-volume-name")

	// Act
	err := asw.MarkVolumeAsAttached(volumeName, volumeSpec, "" /* nodeName */, devicePath)

	// Assert
	if err != nil {
		t.Fatalf("MarkVolumeAsAttached failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsAsw(t, volumeName, true /* shouldExist */, asw)
	verifyVolumeExistsInUnmountedVolumes(t, volumeName, asw)
	verifyVolumeDoesntExistInGloballyMountedVolumes(t, volumeName, asw)
}

// Calls MarkVolumeAsAttached() twice to add the same volume
// Verifies second call doesn't fail
// Verifies newly added volume exists in GetUnmountedVolumes()
// Verifies newly added volume doesn't exist in GetGloballyMountedVolumes()
func Test_MarkVolumeAsAttached_Positive_ExistingVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, plugin := volumetesting.GetTestVolumePluginMgr(t)
	devicePath := "fake/device/path"
	asw := NewActualStateOfWorld("mynode" /* nodeName */, volumePluginMgr)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
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
	generatedVolumeName, _ := volumehelper.GetUniqueVolumeNameFromSpec(plugin, volumeSpec)

	err := asw.MarkVolumeAsAttached(emptyVolumeName, volumeSpec, "" /* nodeName */, devicePath)
	if err != nil {
		t.Fatalf("MarkVolumeAsAttached failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	err = asw.MarkVolumeAsAttached(emptyVolumeName, volumeSpec, "" /* nodeName */, devicePath)

	// Assert
	if err != nil {
		t.Fatalf("MarkVolumeAsAttached failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsAsw(t, generatedVolumeName, true /* shouldExist */, asw)
	verifyVolumeExistsInUnmountedVolumes(t, generatedVolumeName, asw)
	verifyVolumeDoesntExistInGloballyMountedVolumes(t, generatedVolumeName, asw)
}

// Populates data struct with a volume
// Calls AddPodToVolume() to add a pod to the volume
// Verifies volume/pod combo exist using PodExistsInVolume()
func Test_AddPodToVolume_Positive_ExistingVolumeNewNode(t *testing.T) {
	// Arrange
	volumePluginMgr, plugin := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld("mynode" /* nodeName */, volumePluginMgr)
	devicePath := "fake/device/path"

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
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
	generatedVolumeName, err := volumehelper.GetUniqueVolumeNameFromSpec(plugin, volumeSpec)

	err = asw.MarkVolumeAsAttached(emptyVolumeName, volumeSpec, "" /* nodeName */, devicePath)
	if err != nil {
		t.Fatalf("MarkVolumeAsAttached failed. Expected: <no error> Actual: <%v>", err)
	}
	podName := volumehelper.GetUniquePodName(pod)

	mounter, err := plugin.NewMounter(volumeSpec, pod, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("NewMounter failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	err = asw.AddPodToVolume(
		podName, pod.UID, generatedVolumeName, mounter, volumeSpec.Name(), "" /* volumeGidValue */)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsAsw(t, generatedVolumeName, true /* shouldExist */, asw)
	verifyVolumeDoesntExistInUnmountedVolumes(t, generatedVolumeName, asw)
	verifyVolumeDoesntExistInGloballyMountedVolumes(t, generatedVolumeName, asw)
	verifyPodExistsInVolumeAsw(t, podName, generatedVolumeName, "fake/device/path" /* expectedDevicePath */, asw)
}

// Populates data struct with a volume
// Calls AddPodToVolume() twice to add the same pod to the volume
// Verifies volume/pod combo exist using PodExistsInVolume() and the second call
// did not fail.
func Test_AddPodToVolume_Positive_ExistingVolumeExistingNode(t *testing.T) {
	// Arrange
	volumePluginMgr, plugin := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld("mynode" /* nodeName */, volumePluginMgr)
	devicePath := "fake/device/path"

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
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
	generatedVolumeName, err := volumehelper.GetUniqueVolumeNameFromSpec(
		plugin, volumeSpec)

	err = asw.MarkVolumeAsAttached(emptyVolumeName, volumeSpec, "" /* nodeName */, devicePath)
	if err != nil {
		t.Fatalf("MarkVolumeAsAttached failed. Expected: <no error> Actual: <%v>", err)
	}
	podName := volumehelper.GetUniquePodName(pod)

	mounter, err := plugin.NewMounter(volumeSpec, pod, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("NewMounter failed. Expected: <no error> Actual: <%v>", err)
	}

	err = asw.AddPodToVolume(
		podName, pod.UID, generatedVolumeName, mounter, volumeSpec.Name(), "" /* volumeGidValue */)
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	err = asw.AddPodToVolume(
		podName, pod.UID, generatedVolumeName, mounter, volumeSpec.Name(), "" /* volumeGidValue */)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsAsw(t, generatedVolumeName, true /* shouldExist */, asw)
	verifyVolumeDoesntExistInUnmountedVolumes(t, generatedVolumeName, asw)
	verifyVolumeDoesntExistInGloballyMountedVolumes(t, generatedVolumeName, asw)
	verifyPodExistsInVolumeAsw(t, podName, generatedVolumeName, "fake/device/path" /* expectedDevicePath */, asw)
}

// Calls AddPodToVolume() to add pod to empty data stuct
// Verifies call fails with "volume does not exist" error.
func Test_AddPodToVolume_Negative_VolumeDoesntExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld("mynode" /* nodeName */, volumePluginMgr)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
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
	plugin, err := volumePluginMgr.FindPluginBySpec(volumeSpec)
	if err != nil {
		t.Fatalf(
			"volumePluginMgr.FindPluginBySpec failed to find volume plugin for %#v with: %v",
			volumeSpec,
			err)
	}
	volumeName, err := volumehelper.GetUniqueVolumeNameFromSpec(
		plugin, volumeSpec)

	podName := volumehelper.GetUniquePodName(pod)

	mounter, err := plugin.NewMounter(volumeSpec, pod, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("NewMounter failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	err = asw.AddPodToVolume(
		podName, pod.UID, volumeName, mounter, volumeSpec.Name(), "" /* volumeGidValue */)

	// Assert
	if err == nil {
		t.Fatalf("AddPodToVolume did not fail. Expected: <\"no volume with the name ... exists in the list of attached volumes\"> Actual: <no error>")
	}

	verifyVolumeExistsAsw(t, volumeName, false /* shouldExist */, asw)
	verifyVolumeDoesntExistInUnmountedVolumes(t, volumeName, asw)
	verifyVolumeDoesntExistInGloballyMountedVolumes(t, volumeName, asw)
	verifyPodDoesntExistInVolumeAsw(
		t,
		podName,
		volumeName,
		false, /* expectVolumeToExist */
		asw)
}

// Calls MarkVolumeAsAttached() once to add volume
// Calls MarkDeviceAsMounted() to mark volume as globally mounted.
// Verifies newly added volume exists in GetUnmountedVolumes()
// Verifies newly added volume exists in GetGloballyMountedVolumes()
func Test_MarkDeviceAsMounted_Positive_NewVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, plugin := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld("mynode" /* nodeName */, volumePluginMgr)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
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
	devicePath := "fake/device/path"
	generatedVolumeName, err := volumehelper.GetUniqueVolumeNameFromSpec(plugin, volumeSpec)

	err = asw.MarkVolumeAsAttached(emptyVolumeName, volumeSpec, "" /* nodeName */, devicePath)
	if err != nil {
		t.Fatalf("MarkVolumeAsAttached failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	err = asw.MarkDeviceAsMounted(generatedVolumeName)

	// Assert
	if err != nil {
		t.Fatalf("MarkDeviceAsMounted failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsAsw(t, generatedVolumeName, true /* shouldExist */, asw)
	verifyVolumeExistsInUnmountedVolumes(t, generatedVolumeName, asw)
	verifyVolumeExistsInGloballyMountedVolumes(t, generatedVolumeName, asw)
}

func verifyVolumeExistsInGloballyMountedVolumes(
	t *testing.T, expectedVolumeName v1.UniqueVolumeName, asw ActualStateOfWorld) {
	globallyMountedVolumes := asw.GetGloballyMountedVolumes()
	for _, volume := range globallyMountedVolumes {
		if volume.VolumeName == expectedVolumeName {
			return
		}
	}

	t.Fatalf(
		"Could not find volume %v in the list of GloballyMountedVolumes for actual state of world %+v",
		expectedVolumeName,
		globallyMountedVolumes)
}

func verifyVolumeDoesntExistInGloballyMountedVolumes(
	t *testing.T, volumeToCheck v1.UniqueVolumeName, asw ActualStateOfWorld) {
	globallyMountedVolumes := asw.GetGloballyMountedVolumes()
	for _, volume := range globallyMountedVolumes {
		if volume.VolumeName == volumeToCheck {
			t.Fatalf(
				"Found volume %v in the list of GloballyMountedVolumes. Expected it not to exist.",
				volumeToCheck)
		}
	}
}

func verifyVolumeExistsAsw(
	t *testing.T,
	expectedVolumeName v1.UniqueVolumeName,
	shouldExist bool,
	asw ActualStateOfWorld) {
	volumeExists := asw.VolumeExists(expectedVolumeName)
	if shouldExist != volumeExists {
		t.Fatalf(
			"VolumeExists(%q) response incorrect. Expected: <%v> Actual: <%v>",
			expectedVolumeName,
			shouldExist,
			volumeExists)
	}
}

func verifyVolumeExistsInUnmountedVolumes(
	t *testing.T, expectedVolumeName v1.UniqueVolumeName, asw ActualStateOfWorld) {
	unmountedVolumes := asw.GetUnmountedVolumes()
	for _, volume := range unmountedVolumes {
		if volume.VolumeName == expectedVolumeName {
			return
		}
	}

	t.Fatalf(
		"Could not find volume %v in the list of UnmountedVolumes for actual state of world %+v",
		expectedVolumeName,
		unmountedVolumes)
}

func verifyVolumeDoesntExistInUnmountedVolumes(
	t *testing.T, volumeToCheck v1.UniqueVolumeName, asw ActualStateOfWorld) {
	unmountedVolumes := asw.GetUnmountedVolumes()
	for _, volume := range unmountedVolumes {
		if volume.VolumeName == volumeToCheck {
			t.Fatalf(
				"Found volume %v in the list of UnmountedVolumes. Expected it not to exist.",
				volumeToCheck)
		}
	}
}

func verifyPodExistsInVolumeAsw(
	t *testing.T,
	expectedPodName volumetypes.UniquePodName,
	expectedVolumeName v1.UniqueVolumeName,
	expectedDevicePath string,
	asw ActualStateOfWorld) {
	podExistsInVolume, devicePath, err :=
		asw.PodExistsInVolume(expectedPodName, expectedVolumeName)
	if err != nil {
		t.Fatalf(
			"ASW PodExistsInVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	if !podExistsInVolume {
		t.Fatalf(
			"ASW PodExistsInVolume result invalid. Expected: <true> Actual: <%v>",
			podExistsInVolume)
	}

	if devicePath != expectedDevicePath {
		t.Fatalf(
			"Invalid devicePath. Expected: <%q> Actual: <%q> ",
			expectedDevicePath,
			devicePath)
	}
}

func verifyPodDoesntExistInVolumeAsw(
	t *testing.T,
	podToCheck volumetypes.UniquePodName,
	volumeToCheck v1.UniqueVolumeName,
	expectVolumeToExist bool,
	asw ActualStateOfWorld) {
	podExistsInVolume, devicePath, err :=
		asw.PodExistsInVolume(podToCheck, volumeToCheck)
	if !expectVolumeToExist && err == nil {
		t.Fatalf(
			"ASW PodExistsInVolume did not return error. Expected: <error indicating volume does not exist> Actual: <%v>", err)
	}

	if expectVolumeToExist && err != nil {
		t.Fatalf(
			"ASW PodExistsInVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	if podExistsInVolume {
		t.Fatalf(
			"ASW PodExistsInVolume result invalid. Expected: <false> Actual: <%v>",
			podExistsInVolume)
	}

	if devicePath != "" {
		t.Fatalf(
			"Invalid devicePath. Expected: <\"\"> Actual: <%q> ",
			devicePath)
	}
}
