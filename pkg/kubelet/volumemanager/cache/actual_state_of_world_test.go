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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
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
	generatedVolumeName, err := util.GetUniqueVolumeNameFromSpec(plugin, volumeSpec)
	if err != nil {
		t.Fatalf("GetUniqueVolumeNameFromSpec failed. Expected: <no error> Actual: <%v>", err)
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
	generatedVolumeName, err := util.GetUniqueVolumeNameFromSpec(plugin, volumeSpec)
	if err != nil {
		t.Fatalf("GetUniqueVolumeNameFromSpec failed. Expected: <no error> Actual: <%v>", err)
	}

	err = asw.MarkVolumeAsAttached(emptyVolumeName, volumeSpec, "" /* nodeName */, devicePath)
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
	generatedVolumeName, err := util.GetUniqueVolumeNameFromSpec(plugin, volumeSpec)
	if err != nil {
		t.Fatalf("GetUniqueVolumeNameFromSpec failed. Expected: <no error> Actual: <%v>", err)
	}

	err = asw.MarkVolumeAsAttached(emptyVolumeName, volumeSpec, "" /* nodeName */, devicePath)
	if err != nil {
		t.Fatalf("MarkVolumeAsAttached failed. Expected: <no error> Actual: <%v>", err)
	}
	podName := util.GetUniquePodName(pod)

	mounter, err := plugin.NewMounter(volumeSpec, pod, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("NewMounter failed. Expected: <no error> Actual: <%v>", err)
	}

	mapper, err := plugin.NewBlockVolumeMapper(volumeSpec, pod, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("NewBlockVolumeMapper failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	err = asw.AddPodToVolume(
		podName, pod.UID, generatedVolumeName, mounter, mapper, volumeSpec.Name(), "" /* volumeGidValue */, volumeSpec)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsAsw(t, generatedVolumeName, true /* shouldExist */, asw)
	verifyVolumeDoesntExistInUnmountedVolumes(t, generatedVolumeName, asw)
	verifyVolumeDoesntExistInGloballyMountedVolumes(t, generatedVolumeName, asw)
	verifyPodExistsInVolumeAsw(t, podName, generatedVolumeName, "fake/device/path" /* expectedDevicePath */, asw)
	verifyVolumeExistsWithSpecNameInVolumeAsw(t, podName, volumeSpec.Name(), asw)
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
	generatedVolumeName, err := util.GetUniqueVolumeNameFromSpec(
		plugin, volumeSpec)
	if err != nil {
		t.Fatalf("GetUniqueVolumeNameFromSpec failed. Expected: <no error> Actual: <%v>", err)
	}

	err = asw.MarkVolumeAsAttached(emptyVolumeName, volumeSpec, "" /* nodeName */, devicePath)
	if err != nil {
		t.Fatalf("MarkVolumeAsAttached failed. Expected: <no error> Actual: <%v>", err)
	}
	podName := util.GetUniquePodName(pod)

	mounter, err := plugin.NewMounter(volumeSpec, pod, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("NewMounter failed. Expected: <no error> Actual: <%v>", err)
	}

	mapper, err := plugin.NewBlockVolumeMapper(volumeSpec, pod, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("NewBlockVolumeMapper failed. Expected: <no error> Actual: <%v>", err)
	}

	err = asw.AddPodToVolume(
		podName, pod.UID, generatedVolumeName, mounter, mapper, volumeSpec.Name(), "" /* volumeGidValue */, volumeSpec)
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	err = asw.AddPodToVolume(
		podName, pod.UID, generatedVolumeName, mounter, mapper, volumeSpec.Name(), "" /* volumeGidValue */, volumeSpec)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsAsw(t, generatedVolumeName, true /* shouldExist */, asw)
	verifyVolumeDoesntExistInUnmountedVolumes(t, generatedVolumeName, asw)
	verifyVolumeDoesntExistInGloballyMountedVolumes(t, generatedVolumeName, asw)
	verifyPodExistsInVolumeAsw(t, podName, generatedVolumeName, "fake/device/path" /* expectedDevicePath */, asw)
	verifyVolumeExistsWithSpecNameInVolumeAsw(t, podName, volumeSpec.Name(), asw)
}

// Populates data struct with a volume
// Calls AddPodToVolume() twice to add the same pod to the volume
// Verifies volume/pod combo exist using PodExistsInVolume() and the second call
// did not fail.
func Test_AddTwoPodsToVolume_Positive(t *testing.T) {
	// Arrange
	volumePluginMgr, plugin := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld("mynode" /* nodeName */, volumePluginMgr)
	devicePath := "fake/device/path"

	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume-name-1",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	pod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod2",
			UID:  "pod2uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume-name-2",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}
	volumeSpec1 := &volume.Spec{Volume: &pod1.Spec.Volumes[0]}
	volumeSpec2 := &volume.Spec{Volume: &pod2.Spec.Volumes[0]}
	generatedVolumeName1, err := util.GetUniqueVolumeNameFromSpec(
		plugin, volumeSpec1)
	generatedVolumeName2, err := util.GetUniqueVolumeNameFromSpec(
		plugin, volumeSpec2)

	if generatedVolumeName1 != generatedVolumeName2 {
		t.Fatalf(
			"Unique volume names should be the same. unique volume name 1: <%q> unique volume name 2: <%q>, spec1 %v, spec2 %v",
			generatedVolumeName1,
			generatedVolumeName2, volumeSpec1, volumeSpec2)
	}

	err = asw.MarkVolumeAsAttached(generatedVolumeName1, volumeSpec1, "" /* nodeName */, devicePath)
	if err != nil {
		t.Fatalf("MarkVolumeAsAttached failed. Expected: <no error> Actual: <%v>", err)
	}
	podName1 := util.GetUniquePodName(pod1)

	mounter1, err := plugin.NewMounter(volumeSpec1, pod1, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("NewMounter failed. Expected: <no error> Actual: <%v>", err)
	}

	mapper1, err := plugin.NewBlockVolumeMapper(volumeSpec1, pod1, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("NewBlockVolumeMapper failed. Expected: <no error> Actual: <%v>", err)
	}

	err = asw.AddPodToVolume(
		podName1, pod1.UID, generatedVolumeName1, mounter1, mapper1, volumeSpec1.Name(), "" /* volumeGidValue */, volumeSpec1)
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	podName2 := util.GetUniquePodName(pod2)

	mounter2, err := plugin.NewMounter(volumeSpec2, pod2, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("NewMounter failed. Expected: <no error> Actual: <%v>", err)
	}

	mapper2, err := plugin.NewBlockVolumeMapper(volumeSpec2, pod2, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("NewBlockVolumeMapper failed. Expected: <no error> Actual: <%v>", err)
	}

	err = asw.AddPodToVolume(
		podName2, pod2.UID, generatedVolumeName1, mounter2, mapper2, volumeSpec2.Name(), "" /* volumeGidValue */, volumeSpec2)
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsAsw(t, generatedVolumeName1, true /* shouldExist */, asw)
	verifyVolumeDoesntExistInUnmountedVolumes(t, generatedVolumeName1, asw)
	verifyVolumeDoesntExistInGloballyMountedVolumes(t, generatedVolumeName1, asw)
	verifyPodExistsInVolumeAsw(t, podName1, generatedVolumeName1, "fake/device/path" /* expectedDevicePath */, asw)
	verifyVolumeExistsWithSpecNameInVolumeAsw(t, podName1, volumeSpec1.Name(), asw)
	verifyPodExistsInVolumeAsw(t, podName2, generatedVolumeName2, "fake/device/path" /* expectedDevicePath */, asw)
	verifyVolumeExistsWithSpecNameInVolumeAsw(t, podName2, volumeSpec2.Name(), asw)
	verifyVolumeSpecNameInVolumeAsw(t, podName1, []*volume.Spec{volumeSpec1}, asw)
	verifyVolumeSpecNameInVolumeAsw(t, podName2, []*volume.Spec{volumeSpec2}, asw)

}

// Calls AddPodToVolume() to add pod to empty data struct
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

	blockplugin, err := volumePluginMgr.FindMapperPluginBySpec(volumeSpec)
	if err != nil {
		t.Fatalf(
			"volumePluginMgr.FindMapperPluginBySpec failed to find volume plugin for %#v with: %v",
			volumeSpec,
			err)
	}

	volumeName, err := util.GetUniqueVolumeNameFromSpec(
		plugin, volumeSpec)

	podName := util.GetUniquePodName(pod)

	mounter, err := plugin.NewMounter(volumeSpec, pod, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("NewMounter failed. Expected: <no error> Actual: <%v>", err)
	}

	mapper, err := blockplugin.NewBlockVolumeMapper(volumeSpec, pod, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("NewBlockVolumeMapper failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	err = asw.AddPodToVolume(
		podName, pod.UID, volumeName, mounter, mapper, volumeSpec.Name(), "" /* volumeGidValue */, volumeSpec)

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
	verifyVolumeDoesntExistWithSpecNameInVolumeAsw(t, podName, volumeSpec.Name(), asw)
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
	deviceMountPath := "fake/device/mount/path"
	generatedVolumeName, err := util.GetUniqueVolumeNameFromSpec(plugin, volumeSpec)
	if err != nil {
		t.Fatalf("GetUniqueVolumeNameFromSpec failed. Expected: <no error> Actual: <%v>", err)
	}

	err = asw.MarkVolumeAsAttached(emptyVolumeName, volumeSpec, "" /* nodeName */, devicePath)
	if err != nil {
		t.Fatalf("MarkVolumeAsAttached failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	err = asw.MarkDeviceAsMounted(generatedVolumeName, devicePath, deviceMountPath)

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

func verifyVolumeExistsWithSpecNameInVolumeAsw(
	t *testing.T,
	expectedPodName volumetypes.UniquePodName,
	expectedVolumeName string,
	asw ActualStateOfWorld) {
	podExistsInVolume :=
		asw.VolumeExistsWithSpecName(expectedPodName, expectedVolumeName)

	if !podExistsInVolume {
		t.Fatalf(
			"ASW VolumeExistsWithSpecName result invalid. Expected: <true> Actual: <%v>",
			podExistsInVolume)
	}
}

func verifyVolumeDoesntExistWithSpecNameInVolumeAsw(
	t *testing.T,
	podToCheck volumetypes.UniquePodName,
	volumeToCheck string,
	asw ActualStateOfWorld) {
	podExistsInVolume :=
		asw.VolumeExistsWithSpecName(podToCheck, volumeToCheck)

	if podExistsInVolume {
		t.Fatalf(
			"ASW VolumeExistsWithSpecName result invalid. Expected: <false> Actual: <%v>",
			podExistsInVolume)
	}
}

func verifyVolumeSpecNameInVolumeAsw(
	t *testing.T,
	podToCheck volumetypes.UniquePodName,
	volumeSpecs []*volume.Spec,
	asw ActualStateOfWorld) {
	mountedVolumes :=
		asw.GetMountedVolumesForPod(podToCheck)

	for i, volume := range mountedVolumes {
		if volume.InnerVolumeSpecName != volumeSpecs[i].Name() {
			t.Fatalf("Volume spec name does not match Expected: <%q> Actual: <%q>", volumeSpecs[i].Name(), volume.InnerVolumeSpecName)
		}
	}
}
