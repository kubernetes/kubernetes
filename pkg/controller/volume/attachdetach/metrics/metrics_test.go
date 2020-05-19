/*
Copyright 2018 The Kubernetes Authors.

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

package metrics

import (
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	csitrans "k8s.io/csi-translation-lib"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/attachdetach/testing"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/types"
)

func TestVolumesInUseMetricCollection(t *testing.T) {
	fakeVolumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	fakeClient := &fake.Clientset{}

	fakeInformerFactory := informers.NewSharedInformerFactory(fakeClient, controller.NoResyncPeriodFunc())
	fakePodInformer := fakeInformerFactory.Core().V1().Pods()
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "metric-test-pod",
			UID:       "metric-test-pod-uid",
			Namespace: "metric-test",
		},
		Spec: v1.PodSpec{
			NodeName: "metric-test-host",
			Volumes: []v1.Volume{
				{
					Name: "metric-test-volume-name",
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "metric-test-pvc",
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
	pvcInformer := fakeInformerFactory.Core().V1().PersistentVolumeClaims()
	pvInformer := fakeInformerFactory.Core().V1().PersistentVolumes()

	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "metric-test-pvc",
			Namespace: "metric-test",
			UID:       "metric-test-pvc-1",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany, v1.ReadWriteOnce},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("2G"),
				},
			},
			VolumeName: "test-metric-pv-1",
		},
		Status: v1.PersistentVolumeClaimStatus{
			Phase: v1.ClaimBound,
		},
	}
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			UID:  "test-metric-pv-1",
			Name: "test-metric-pv-1",
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("5G"),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce, v1.ReadOnlyMany},
			// this one we're pretending is already bound
			ClaimRef: &v1.ObjectReference{UID: "metric-test-pvc-1", Namespace: "metric-test"},
		},
	}
	pvcInformer.Informer().GetStore().Add(pvc)
	pvInformer.Informer().GetStore().Add(pv)
	pvcLister := pvcInformer.Lister()
	pvLister := pvInformer.Lister()

	csiTranslator := csitrans.New()
	metricCollector := newAttachDetachStateCollector(
		pvcLister,
		fakePodInformer.Lister(),
		pvLister,
		nil,
		nil,
		fakeVolumePluginMgr,
		csimigration.NewPluginManager(csiTranslator),
		csiTranslator)
	nodeUseMap := metricCollector.getVolumeInUseCount()
	if len(nodeUseMap) < 1 {
		t.Errorf("Expected one volume in use got %d", len(nodeUseMap))
	}
	testNodeMetric := nodeUseMap["metric-test-host"]
	pluginUseCount, ok := testNodeMetric["fake-plugin"]
	if !ok {
		t.Errorf("Expected fake plugin pvc got nothing")
	}

	if pluginUseCount < 1 {
		t.Errorf("Expected at least in-use volume metric got %d", pluginUseCount)
	}
}

func TestTotalVolumesMetricCollection(t *testing.T) {
	fakeVolumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(fakeVolumePluginMgr)
	asw := cache.NewActualStateOfWorld(fakeVolumePluginMgr)
	podName := "pod-uid"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := k8stypes.NodeName("node-name")

	dsw.AddNode(nodeName, false)
	_, err := dsw.AddPod(types.UniquePodName(podName), controllervolumetesting.NewPod(podName, podName), volumeSpec, nodeName)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	asw.AddVolumeNode(volumeName, volumeSpec, nodeName, "", true)

	csiTranslator := csitrans.New()
	metricCollector := newAttachDetachStateCollector(
		nil,
		nil,
		nil,
		asw,
		dsw,
		fakeVolumePluginMgr,
		csimigration.NewPluginManager(csiTranslator),
		csiTranslator)

	totalVolumesMap := metricCollector.getTotalVolumesCount()
	if len(totalVolumesMap) != 2 {
		t.Errorf("Expected 2 states, got %d", len(totalVolumesMap))
	}

	dswCount, ok := totalVolumesMap["desired_state_of_world"]
	if !ok {
		t.Errorf("Expected desired_state_of_world, got nothing")
	}

	fakePluginCount := dswCount["fake-plugin"]
	if fakePluginCount != 1 {
		t.Errorf("Expected 1 fake-plugin volume in DesiredStateOfWorld, got %d", fakePluginCount)
	}

	aswCount, ok := totalVolumesMap["actual_state_of_world"]
	if !ok {
		t.Errorf("Expected actual_state_of_world, got nothing")
	}

	fakePluginCount = aswCount["fake-plugin"]
	if fakePluginCount != 1 {
		t.Errorf("Expected 1 fake-plugin volume in ActualStateOfWorld, got %d", fakePluginCount)
	}
}
