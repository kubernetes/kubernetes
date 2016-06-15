/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

// Calls AddVolume() once to add volume
// Verifies newly added volume exists in GetAttachedVolumes()
func Test_AddVolume_Positive_NewVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld("mynode" /* nodeName */, volumePluginMgr)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "volume-name",
					VolumeSource: api.VolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}
	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}

	// Act
	generatedVolumeName, err := asw.AddVolume(volumeSpec)

	// Assert
	if err != nil {
		t.Fatalf("AddVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsInAttachedVolumes(t, generatedVolumeName, asw)
}

// Calls AddVolume() twice to add the same volume
// Verifies newly added volume exists in GetAttachedVolumes() and second call
// doesn't fail
func Test_AddVolume_Positive_ExistingVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld("mynode" /* nodeName */, volumePluginMgr)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "volume-name",
					VolumeSource: api.VolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	generatedVolumeName, err := asw.AddVolume(volumeSpec)
	if err != nil {
		t.Fatalf("AddVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	generatedVolumeName, err = asw.AddVolume(volumeSpec)

	// Assert
	if err != nil {
		t.Fatalf("AddVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsInAttachedVolumes(t, generatedVolumeName, asw)
}

// Populates data struct with a volume
// Calls AddPodToVolume() to add a pod to the volume
// Verifies volume/pod combo exist using PodExistsInVolume()
func Test_AddPodToVolume_Positive_ExistingVolumeNewNode(t *testing.T) {
	// Arrange
	volumePluginMgr, plugin := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld("mynode" /* nodeName */, volumePluginMgr)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "volume-name",
					VolumeSource: api.VolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	volumeName, err := volumehelper.GetUniqueVolumeNameFromSpec(
		plugin, volumeSpec)

	generatedVolumeName, err := asw.AddVolume(volumeSpec)
	if err != nil {
		t.Fatalf("AddVolume failed. Expected: <no error> Actual: <%v>", err)
	}
	podName := volumehelper.GetUniquePodName(pod)

	mounter, err := plugin.NewMounter(volumeSpec, pod, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("NewUnmounter failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	err = asw.AddPodToVolume(
		podName, pod.UID, volumeName, mounter, volumeSpec.Name(), "" /* volumeGidValue */)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsInAttachedVolumes(t, generatedVolumeName, asw)
	verifyPodExistsInVolumeAsw(t, podName, generatedVolumeName, asw)
}

// Populates data struct with a volume
// Calls AddPodToVolume() twice to add the same pod to the volume
// Verifies volume/pod combo exist using PodExistsInVolume() and the second call
// did not fail.
func Test_AddPodToVolume_Positive_ExistingVolumeExistingNode(t *testing.T) {
	// Arrange
	volumePluginMgr, plugin := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld("mynode" /* nodeName */, volumePluginMgr)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "volume-name",
					VolumeSource: api.VolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	volumeName, err := volumehelper.GetUniqueVolumeNameFromSpec(
		plugin, volumeSpec)

	generatedVolumeName, err := asw.AddVolume(volumeSpec)
	if err != nil {
		t.Fatalf("AddVolume failed. Expected: <no error> Actual: <%v>", err)
	}
	podName := volumehelper.GetUniquePodName(pod)

	mounter, err := plugin.NewMounter(volumeSpec, pod, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("NewUnmounter failed. Expected: <no error> Actual: <%v>", err)
	}

	err = asw.AddPodToVolume(
		podName, pod.UID, volumeName, mounter, volumeSpec.Name(), "" /* volumeGidValue */)
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	err = asw.AddPodToVolume(
		podName, pod.UID, volumeName, mounter, volumeSpec.Name(), "" /* volumeGidValue */)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsInAttachedVolumes(t, generatedVolumeName, asw)
	verifyPodExistsInVolumeAsw(t, podName, generatedVolumeName, asw)
}

// Calls AddPodToVolume() to add pod to empty data stuct
// Verifies call fails with "volume does not exist" error.
func Test_AddPodToVolume_Negative_VolumeDoesntExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld("mynode" /* nodeName */, volumePluginMgr)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "volume-name",
					VolumeSource: api.VolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
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
		t.Fatalf("NewUnmounter failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	err = asw.AddPodToVolume(
		podName, pod.UID, volumeName, mounter, volumeSpec.Name(), "" /* volumeGidValue */)

	// Assert
	if err == nil {
		t.Fatalf("AddPodToVolume did not fail. Expected: <\"no volume with the name ... exists in the list of attached volumes\"> Actual: <no error>")
	}

	verifyVolumeDoesntExistInAttachedVolumes(t, volumeName, asw)
	verifyPodDoesntExistInVolumeAsw(
		t,
		podName,
		volumeName,
		false, /* expectVolumeToExist */
		asw)
}

func verifyVolumeExistsInAttachedVolumes(
	t *testing.T, expectedVolumeName api.UniqueVolumeName, asw ActualStateOfWorld) {
	attachedVolumes := asw.GetAttachedVolumes()
	for _, volume := range attachedVolumes {
		if volume.VolumeName == expectedVolumeName {
			return
		}
	}

	t.Fatalf(
		"Could not find volume %v in the list of attached volumes for actual state of world %+v",
		expectedVolumeName,
		attachedVolumes)
}

func verifyVolumeDoesntExistInAttachedVolumes(
	t *testing.T, volumeToCheck api.UniqueVolumeName, asw ActualStateOfWorld) {
	attachedVolumes := asw.GetAttachedVolumes()
	for _, volume := range attachedVolumes {
		if volume.VolumeName == volumeToCheck {
			t.Fatalf(
				"Found volume %v in the list of attached volumes. Expected it not to exist.",
				volumeToCheck)
		}
	}
}

func verifyPodExistsInVolumeAsw(
	t *testing.T,
	expectedPodName volumetypes.UniquePodName,
	expectedVolumeName api.UniqueVolumeName,
	asw ActualStateOfWorld) {
	podExistsInVolume, err :=
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
}

func verifyPodDoesntExistInVolumeAsw(
	t *testing.T,
	podToCheck volumetypes.UniquePodName,
	volumeToCheck api.UniqueVolumeName,
	expectVolumeToExist bool,
	asw ActualStateOfWorld) {
	podExistsInVolume, err :=
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
}
