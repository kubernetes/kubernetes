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

package populator

import (
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	csitrans "k8s.io/csi-translation-lib"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
)

func TestFindAndAddActivePods_FindAndRemoveDeletedPods(t *testing.T) {
	fakeVolumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	fakeClient := &fake.Clientset{}

	fakeInformerFactory := informers.NewSharedInformerFactory(fakeClient, controller.NoResyncPeriodFunc())
	fakePodInformer := fakeInformerFactory.Core().V1().Pods()

	fakesDSW := cache.NewDesiredStateOfWorld(fakeVolumePluginMgr)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dswp-test-pod",
			UID:       "dswp-test-pod-uid",
			Namespace: "dswp-test",
		},
		Spec: v1.PodSpec{
			NodeName: "dswp-test-host",
			Volumes: []v1.Volume{
				{
					Name: "dswp-test-volume-name",
					VolumeSource: v1.VolumeSource{
						RBD: &v1.RBDVolumeSource{
							RBDImage: "dswp-test-fake-device",
						},
					},
				},
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodPhase("Running"),
		},
	}

	fakePodInformer.Informer().GetStore().Add(pod)

	podName := util.GetUniquePodName(pod)

	generatedVolumeName := "fake-plugin/" + pod.Spec.Volumes[0].RBD.RBDImage

	pvcLister := fakeInformerFactory.Core().V1().PersistentVolumeClaims().Lister()
	pvLister := fakeInformerFactory.Core().V1().PersistentVolumes().Lister()

	csiTranslator := csitrans.New()
	dswp := &desiredStateOfWorldPopulator{
		loopSleepDuration:        100 * time.Millisecond,
		listPodsRetryDuration:    3 * time.Second,
		desiredStateOfWorld:      fakesDSW,
		volumePluginMgr:          fakeVolumePluginMgr,
		podLister:                fakePodInformer.Lister(),
		pvcLister:                pvcLister,
		pvLister:                 pvLister,
		csiMigratedPluginManager: csimigration.NewPluginManager(csiTranslator),
		intreeToCSITranslator:    csiTranslator,
	}

	//add the given node to the list of nodes managed by dsw
	dswp.desiredStateOfWorld.AddNode(k8stypes.NodeName(pod.Spec.NodeName))
	logger, _ := ktesting.NewTestContext(t)
	dswp.findAndAddActivePods(logger)

	expectedVolumeName := v1.UniqueVolumeName(generatedVolumeName)

	//check if the given volume referenced by the pod is added to dsw
	volumeExists := dswp.desiredStateOfWorld.VolumeExists(expectedVolumeName, k8stypes.NodeName(pod.Spec.NodeName))
	if !volumeExists {
		t.Fatalf(
			"VolumeExists(%q) failed. Expected: <true> Actual: <%v>",
			expectedVolumeName,
			volumeExists)
	}

	//delete the pod and volume manually
	dswp.desiredStateOfWorld.DeletePod(podName, expectedVolumeName, k8stypes.NodeName(pod.Spec.NodeName))

	//check if the given volume referenced by the pod still exists in dsw
	volumeExists = dswp.desiredStateOfWorld.VolumeExists(expectedVolumeName, k8stypes.NodeName(pod.Spec.NodeName))
	if volumeExists {
		t.Fatalf(
			"VolumeExists(%q) failed. Expected: <false> Actual: <%v>",
			expectedVolumeName,
			volumeExists)
	}

	//add pod and volume again
	dswp.findAndAddActivePods(logger)

	//check if the given volume referenced by the pod is added to dsw for the second time
	volumeExists = dswp.desiredStateOfWorld.VolumeExists(expectedVolumeName, k8stypes.NodeName(pod.Spec.NodeName))
	if !volumeExists {
		t.Fatalf(
			"VolumeExists(%q) failed. Expected: <true> Actual: <%v>",
			expectedVolumeName,
			volumeExists)
	}

	fakePodInformer.Informer().GetStore().Delete(pod)
	dswp.findAndRemoveDeletedPods(logger)
	//check if the given volume referenced by the pod still exists in dsw
	volumeExists = dswp.desiredStateOfWorld.VolumeExists(expectedVolumeName, k8stypes.NodeName(pod.Spec.NodeName))
	if volumeExists {
		t.Fatalf(
			"VolumeExists(%q) failed. Expected: <false> Actual: <%v>",
			expectedVolumeName,
			volumeExists)
	}

}

func TestFindAndRemoveNonattachableVolumes(t *testing.T) {
	fakeVolumePluginMgr, fakeVolumePlugin := volumetesting.GetTestVolumePluginMgr(t)
	fakeClient := &fake.Clientset{}

	fakeInformerFactory := informers.NewSharedInformerFactory(fakeClient, controller.NoResyncPeriodFunc())
	fakePodInformer := fakeInformerFactory.Core().V1().Pods()

	fakesDSW := cache.NewDesiredStateOfWorld(fakeVolumePluginMgr)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dswp-test-pod",
			UID:       "dswp-test-pod-uid",
			Namespace: "dswp-test",
		},
		Spec: v1.PodSpec{
			NodeName: "dswp-test-host",
			Volumes: []v1.Volume{
				{
					Name: "dswp-test-volume-name",
					VolumeSource: v1.VolumeSource{
						CSI: &v1.CSIVolumeSource{
							Driver: "dswp-test-fake-csi-driver",
						},
					},
				},
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodPhase("Running"),
		},
	}

	fakePodInformer.Informer().GetStore().Add(pod)

	generatedVolumeName := "fake-plugin/dswp-test-fake-csi-driver"
	pvcLister := fakeInformerFactory.Core().V1().PersistentVolumeClaims().Lister()
	pvLister := fakeInformerFactory.Core().V1().PersistentVolumes().Lister()

	csiTranslator := csitrans.New()
	dswp := &desiredStateOfWorldPopulator{
		loopSleepDuration:        100 * time.Millisecond,
		listPodsRetryDuration:    3 * time.Second,
		desiredStateOfWorld:      fakesDSW,
		volumePluginMgr:          fakeVolumePluginMgr,
		podLister:                fakePodInformer.Lister(),
		pvcLister:                pvcLister,
		pvLister:                 pvLister,
		csiMigratedPluginManager: csimigration.NewPluginManager(csiTranslator),
		intreeToCSITranslator:    csiTranslator,
	}

	//add the given node to the list of nodes managed by dsw
	dswp.desiredStateOfWorld.AddNode(k8stypes.NodeName(pod.Spec.NodeName))
	logger, _ := ktesting.NewTestContext(t)
	dswp.findAndAddActivePods(logger)

	expectedVolumeName := v1.UniqueVolumeName(generatedVolumeName)

	//check if the given volume referenced by the pod is added to dsw
	volumeExists := dswp.desiredStateOfWorld.VolumeExists(expectedVolumeName, k8stypes.NodeName(pod.Spec.NodeName))
	if !volumeExists {
		t.Fatalf(
			"VolumeExists(%q) failed. Expected: <true> Actual: <%v>",
			expectedVolumeName,
			volumeExists)
	}

	// Change the CSI volume plugin attachability
	fakeVolumePlugin.NonAttachable = true

	dswp.findAndRemoveDeletedPods(logger)

	// The volume should not exist after it becomes non-attachable
	volumeExists = dswp.desiredStateOfWorld.VolumeExists(expectedVolumeName, k8stypes.NodeName(pod.Spec.NodeName))
	if volumeExists {
		t.Fatalf(
			"VolumeExists(%q) failed. Expected: <false> Actual: <%v>",
			expectedVolumeName,
			volumeExists)
	}
}

func TestFindAndAddActivePods_SkipsTerminatingPodsOnUnhealthyNode(t *testing.T) {
	fakeVolumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	fakeClient := &fake.Clientset{}

	fakeInformerFactory := informers.NewSharedInformerFactory(fakeClient, controller.NoResyncPeriodFunc())
	fakePodInformer := fakeInformerFactory.Core().V1().Pods()
	fakeNodeInformer := fakeInformerFactory.Core().V1().Nodes()

	fakesDSW := cache.NewDesiredStateOfWorld(fakeVolumePluginMgr)

	now := metav1.Now()
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "dswp-test-pod",
			UID:               "dswp-test-pod-uid",
			Namespace:         "dswp-test",
			DeletionTimestamp: &now,
		},
		Spec: v1.PodSpec{
			NodeName: "dswp-test-host",
			Volumes: []v1.Volume{
				{
					Name: "dswp-test-volume-name",
					VolumeSource: v1.VolumeSource{
						RBD: &v1.RBDVolumeSource{
							RBDImage: "dswp-test-fake-device",
						},
					},
				},
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodPhase("Running"),
			ContainerStatuses: []v1.ContainerStatus{
				{
					Name: "container",
					State: v1.ContainerState{
						Running: &v1.ContainerStateRunning{},
					},
				},
			},
		},
	}

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "dswp-test-host",
		},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionFalse,
				},
			},
		},
	}

	fakePodInformer.Informer().GetStore().Add(pod)
	fakeNodeInformer.Informer().GetStore().Add(node)

	generatedVolumeName := "fake-plugin/" + pod.Spec.Volumes[0].RBD.RBDImage
	expectedVolumeName := v1.UniqueVolumeName(generatedVolumeName)

	pvcLister := fakeInformerFactory.Core().V1().PersistentVolumeClaims().Lister()
	pvLister := fakeInformerFactory.Core().V1().PersistentVolumes().Lister()

	csiTranslator := csitrans.New()
	dswp := &desiredStateOfWorldPopulator{
		loopSleepDuration:        100 * time.Millisecond,
		listPodsRetryDuration:    3 * time.Second,
		desiredStateOfWorld:      fakesDSW,
		volumePluginMgr:          fakeVolumePluginMgr,
		podLister:                fakePodInformer.Lister(),
		pvcLister:                pvcLister,
		pvLister:                 pvLister,
		nodeLister:               fakeNodeInformer.Lister(),
		csiMigratedPluginManager: csimigration.NewPluginManager(csiTranslator),
		intreeToCSITranslator:    csiTranslator,
	}

	dswp.desiredStateOfWorld.AddNode(k8stypes.NodeName(pod.Spec.NodeName))
	logger, _ := ktesting.NewTestContext(t)
	dswp.findAndAddActivePods(logger)

	// Since pod is terminating on an unhealthy node, volume should NOT be added to DSW
	volumeExists := dswp.desiredStateOfWorld.VolumeExists(expectedVolumeName, k8stypes.NodeName(pod.Spec.NodeName))
	if volumeExists {
		t.Fatalf("VolumeExists(%q) should be false for terminating pod on unhealthy node, got true", expectedVolumeName)
	}

	// Now set node to Ready
	node.Status.Conditions[0].Status = v1.ConditionTrue
	fakeNodeInformer.Informer().GetStore().Update(node)

	// Volume should now be added to DSW because node is healthy
	dswp.findAndAddActivePods(logger)
	volumeExists = dswp.desiredStateOfWorld.VolumeExists(expectedVolumeName, k8stypes.NodeName(pod.Spec.NodeName))
	if !volumeExists {
		t.Fatalf("VolumeExists(%q) should be true for terminating pod on healthy node, got false", expectedVolumeName)
	}
}

func TestFindAndRemoveDeletedPods_RemovesTerminatingPodsOnUnhealthyNode(t *testing.T) {
	fakeVolumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	fakeClient := &fake.Clientset{}

	fakeInformerFactory := informers.NewSharedInformerFactory(fakeClient, controller.NoResyncPeriodFunc())
	fakePodInformer := fakeInformerFactory.Core().V1().Pods()
	fakeNodeInformer := fakeInformerFactory.Core().V1().Nodes()

	fakesDSW := cache.NewDesiredStateOfWorld(fakeVolumePluginMgr)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dswp-test-pod",
			UID:       "dswp-test-pod-uid",
			Namespace: "dswp-test",
		},
		Spec: v1.PodSpec{
			NodeName: "dswp-test-host",
			Volumes: []v1.Volume{
				{
					Name: "dswp-test-volume-name",
					VolumeSource: v1.VolumeSource{
						RBD: &v1.RBDVolumeSource{
							RBDImage: "dswp-test-fake-device",
						},
					},
				},
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodPhase("Running"),
			ContainerStatuses: []v1.ContainerStatus{
				{
					Name: "container",
					State: v1.ContainerState{
						Running: &v1.ContainerStateRunning{},
					},
				},
			},
		},
	}

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "dswp-test-host",
		},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionTrue,
				},
			},
		},
	}

	fakePodInformer.Informer().GetStore().Add(pod)
	fakeNodeInformer.Informer().GetStore().Add(node)

	generatedVolumeName := "fake-plugin/" + pod.Spec.Volumes[0].RBD.RBDImage
	expectedVolumeName := v1.UniqueVolumeName(generatedVolumeName)

	pvcLister := fakeInformerFactory.Core().V1().PersistentVolumeClaims().Lister()
	pvLister := fakeInformerFactory.Core().V1().PersistentVolumes().Lister()

	csiTranslator := csitrans.New()
	dswp := &desiredStateOfWorldPopulator{
		loopSleepDuration:        100 * time.Millisecond,
		listPodsRetryDuration:    3 * time.Second,
		desiredStateOfWorld:      fakesDSW,
		volumePluginMgr:          fakeVolumePluginMgr,
		podLister:                fakePodInformer.Lister(),
		pvcLister:                pvcLister,
		pvLister:                 pvLister,
		nodeLister:               fakeNodeInformer.Lister(),
		csiMigratedPluginManager: csimigration.NewPluginManager(csiTranslator),
		intreeToCSITranslator:    csiTranslator,
	}

	dswp.desiredStateOfWorld.AddNode(k8stypes.NodeName(pod.Spec.NodeName))
	logger, _ := ktesting.NewTestContext(t)
	dswp.findAndAddActivePods(logger)

	// Verify volume exists in DSW
	volumeExists := dswp.desiredStateOfWorld.VolumeExists(expectedVolumeName, k8stypes.NodeName(pod.Spec.NodeName))
	if !volumeExists {
		t.Fatalf("VolumeExists(%q) should be true, got false", expectedVolumeName)
	}

	// Now node crashes: node becomes NotReady, pod gets DeletionTimestamp
	node.Status.Conditions[0].Status = v1.ConditionFalse
	fakeNodeInformer.Informer().GetStore().Update(node)

	now := metav1.Now()
	pod.DeletionTimestamp = &now
	fakePodInformer.Informer().GetStore().Update(pod)

	// Running findAndRemoveDeletedPods should remove the terminating pod on the unhealthy node from DSW
	dswp.findAndRemoveDeletedPods(logger)

	volumeExists = dswp.desiredStateOfWorld.VolumeExists(expectedVolumeName, k8stypes.NodeName(pod.Spec.NodeName))
	if volumeExists {
		t.Fatalf("VolumeExists(%q) should be false after pod is terminating on unhealthy node, got true", expectedVolumeName)
	}
}
