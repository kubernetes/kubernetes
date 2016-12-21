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

package operationexecutor

import (
	"k8s.io/client-go/_vendor/github.com/pborman/uuid"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"strconv"
	"testing"
	"time"
)

var _ OperationGenerator = &mockOperationGenerator{}

func TestOperationExecutor_MountVolume_ParallelMountForNonAttachablePlugins(t *testing.T) {
	// Arrange
	oe := NewOperationExecutor(newMockOperationGenerator())

	numVolumesToMount := 2
	volumesToMount := make([]VolumeToMount, numVolumesToMount)
	secretName := "secret-volume"
	volumeName := v1.UniqueVolumeName(secretName)

	// Act
	for i := range volumesToMount {
		podName := "pod-" + strconv.Itoa((i + 1))
		pod := getTestPodWithSecret(podName, secretName)
		volumesToMount[i] = VolumeToMount{
			Pod:                pod,
			VolumeName:         volumeName,
			PluginIsAttachable: false, /*this field determines whether the plugin is attachable*/
			ReportedInUse:      true,
		}
		oe.MountVolume(0 /*waitForAttachTimeOut*/, volumesToMount[i], nil /*actualStateOfWorldMounterUpdater*/)
	}
	// This sleep is to ensure that we don't assert before the mount operations have made it to
	// the operations data structure
	time.Sleep(1 * time.Second)

	// Assert
	if len(oe.GetNestedPendingOperations().GetOperations()) != numVolumesToMount {
		t.Fatalf("Unable to mount in parallel for a non-attachable plugin")
	}
}

func TestOperationExecutor_MountVolume_ParallelMountForAttachablePlugins(t *testing.T) {
	// Arrange
	oe := NewOperationExecutor(newMockOperationGenerator())

	numVolumesToMount := 2
	volumesToMount := make([]VolumeToMount, numVolumesToMount)
	pdName := "pd-volume"
	volumeName := v1.UniqueVolumeName(pdName)

	// Act
	for i := range volumesToMount {
		podName := "pod-" + string((i + 1))
		pod := getTestPodWithGCEPD(podName, pdName)
		volumesToMount[i] = VolumeToMount{
			Pod:                pod,
			VolumeName:         volumeName,
			PluginIsAttachable: true, /*this field determines whether the plugin is attachable*/
			ReportedInUse:      true,
		}
		oe.MountVolume(0 /*waitForAttachTimeout*/, volumesToMount[i], nil /*actualStateOfWorldMounterUpdater*/)
	}
	// This sleep is to ensure that we don't assert before the mount operations have made it to
	// the operations data structure
	time.Sleep(1 * time.Second)

	// Assert
	if len(oe.GetNestedPendingOperations().GetOperations()) != 1 {
		t.Fatalf("Parallel mount shouldn't happen for attachable plugins")
	}
}

type mockOperationGenerator struct {
}

func newMockOperationGenerator() OperationGenerator {
	return &mockOperationGenerator{}
}

func (mopg *mockOperationGenerator) GenerateMountVolumeFunc(waitForAttachTimeout time.Duration, volumeToMount VolumeToMount, actualStateOfWorldMounterUpdater ActualStateOfWorldMounterUpdater) (func() error, error) {
	return func() error {
		// Mocking the behavior of mount to take a long time to
		// assert if mounts can be done in parallel
		time.Sleep(10 * time.Second)
		return nil
	}, nil
}
func (mopg *mockOperationGenerator) GenerateUnmountVolumeFunc(volumeToUnmount MountedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater) (func() error, error) {
	return func() error { return nil }, nil
}
func (mopg *mockOperationGenerator) GenerateAttachVolumeFunc(volumeToAttach VolumeToAttach, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error) {
	return func() error { return nil }, nil
}
func (mopg *mockOperationGenerator) GenerateDetachVolumeFunc(volumeToDetach AttachedVolume, verifySafeToDetach bool, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error) {
	return func() error { return nil }, nil
}
func (mopg *mockOperationGenerator) GenerateVolumesAreAttachedFunc(attachedVolumes []AttachedVolume, nodeName types.NodeName, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error) {
	return func() error { return nil }, nil
}
func (mopg *mockOperationGenerator) GenerateUnmountDeviceFunc(deviceToDetach AttachedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, mounter mount.Interface) (func() error, error) {
	return func() error { return nil }, nil
}
func (mopg *mockOperationGenerator) GenerateVerifyControllerAttachedVolumeFunc(volumeToMount VolumeToMount, nodeName types.NodeName, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error) {
	return func() error { return nil }, nil
}

func getTestPodWithSecret(podName, secretName string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: podName,
			UID:  types.UID(podName),
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: secretName,
					VolumeSource: v1.VolumeSource{
						Secret: &v1.SecretVolumeSource{
							SecretName: secretName,
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  "secret-volume-test",
					Image: "gcr.io/google_containers/mounttest:0.7",
					Args: []string{
						"--file_content=/etc/secret-volume/data-1",
						"--file_mode=/etc/secret-volume/data-1"},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      secretName,
							MountPath: "/data",
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
}

func getTestPodWithGCEPD(podName, pdName string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: podName,
			UID:  types.UID(podName + string(uuid.New())),
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: pdName,
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName:   pdName,
							FSType:   "ext4",
							ReadOnly: false,
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  "pd-volume-test",
					Image: "gcr.io/google_containers/mounttest:0.7",
					Args: []string{
						"--file_content=/etc/pd-volume/data-1",
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      pdName,
							MountPath: "/data",
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
}
