/*
Copyright 2023 The Kubernetes Authors.

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

package reconciler

import (
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/hostutil"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	"k8s.io/mount-utils"
)

func TestReconcileWithUpdateReconstructedFromAPIServer(t *testing.T) {
	// Calls Run() with two reconstructed volumes.
	// Verifies the devicePaths + volume attachability are reconstructed from node.status.

	// Arrange
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: string(nodeName),
		},
		Status: v1.NodeStatus{
			VolumesAttached: []v1.AttachedVolume{
				{
					Name:       "fake-plugin/fake-device1",
					DevicePath: "fake/path",
				},
			},
		},
	}
	volumePluginMgr, fakePlugin := volumetesting.GetTestKubeletVolumePluginMgrWithNode(t, node)
	seLinuxTranslator := util.NewFakeSELinuxLabelTranslator()
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr, seLinuxTranslator)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createTestClient()
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		kubeClient,
		volumePluginMgr,
		fakeRecorder,
		fakeHandler))
	rc := NewReconciler(
		kubeClient,
		true, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		hasAddedPods,
		oex,
		mount.NewFakeMounter(nil),
		hostutil.NewFakeHostUtil(nil),
		volumePluginMgr,
		kubeletPodsDir)
	reconciler := rc.(*reconciler)

	// The pod has two volumes, fake-device1 is attachable, fake-device2 is not.
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
				{
					Name: "volume-name2",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device2",
						},
					},
				},
			},
		},
	}

	volumeSpec1 := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	volumeName1 := util.GetUniqueVolumeName(fakePlugin.GetPluginName(), "fake-device1")
	volumeSpec2 := &volume.Spec{Volume: &pod.Spec.Volumes[1]}
	volumeName2 := util.GetUniqueVolumeName(fakePlugin.GetPluginName(), "fake-device2")

	assert.NoError(t, asw.AddAttachUncertainReconstructedVolume(volumeName1, volumeSpec1, nodeName, ""))
	assert.NoError(t, asw.MarkDeviceAsUncertain(volumeName1, "/dev/badly/reconstructed", "/var/lib/kubelet/plugins/global1", ""))
	assert.NoError(t, asw.AddAttachUncertainReconstructedVolume(volumeName2, volumeSpec2, nodeName, ""))
	assert.NoError(t, asw.MarkDeviceAsUncertain(volumeName2, "/dev/reconstructed", "/var/lib/kubelet/plugins/global2", ""))

	assert.False(t, reconciler.StatesHasBeenSynced())

	reconciler.volumesNeedUpdateFromNodeStatus = append(reconciler.volumesNeedUpdateFromNodeStatus, volumeName1, volumeName2)
	// Act - run reconcile loop just once.
	// "volumesNeedUpdateFromNodeStatus" is not empty, so no unmount will be triggered.
	reconciler.reconcileNew()

	// Assert
	assert.True(t, reconciler.StatesHasBeenSynced())
	assert.Empty(t, reconciler.volumesNeedUpdateFromNodeStatus)

	attachedVolumes := asw.GetAttachedVolumes()
	assert.Equalf(t, len(attachedVolumes), 2, "two volumes in ASW expected")
	for _, vol := range attachedVolumes {
		if vol.VolumeName == volumeName1 {
			// devicePath + attachability must have been updated from node.status
			assert.True(t, vol.PluginIsAttachable)
			assert.Equal(t, vol.DevicePath, "fake/path")
		}
		if vol.VolumeName == volumeName2 {
			// only attachability was updated from node.status
			assert.False(t, vol.PluginIsAttachable)
			assert.Equal(t, vol.DevicePath, "/dev/reconstructed")
		}
	}
}
