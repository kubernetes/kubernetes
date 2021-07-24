/*
Copyright 2014 The Kubernetes Authors.

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

package volume

import (
	"context"
	"fmt"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
	"testing"
)

func TestLocalPersistentVolume(t *testing.T) {
	klog.V(2).Infof("TestLocalPersistentVolume started: Pods sharing a single local PV")
	_, s, closeFn := framework.RunAnAPIServer(nil)
	defer closeFn()

	ns := framework.CreateTestingNamespace("local-pv", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	testClient, ctrl, informers, watchPV, watchPVC := createClients(ns, t, s, defaultSyncPeriod)
	defer watchPV.Stop()
	defer watchPVC.Stop()

	// NOTE: This test cannot run in parallel, because it is creating and deleting
	// non-namespaced objects (PersistenceVolumes).
	defer testClient.CoreV1().PersistentVolumes().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})

	stopCh := make(chan struct{})
	informers.Start(stopCh)
	go ctrl.Run(stopCh)
	defer close(stopCh)
	namespace := "local-pv"
	pv := createLocalPV("local-pv")
	pvc := createLocalPVC("local-pvc", namespace)

	_, err := testClient.CoreV1().PersistentVolumes().Create(context.TODO(), pv, metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Failed to create PersistentVolume: %v", err)
	}
	klog.V(2).Infof("TestLocalPersistentVolume pvc created")

	_, err = testClient.CoreV1().PersistentVolumeClaims(namespace).Create(context.TODO(), pvc, metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Failed to create PersistentVolumeClaim: %v", err)
	}

	klog.V(2).Infof("TestLocalPersistentVolume pvc created")
	var createPods []*v1.Pod

	for i := 0; i < 50; i++ {
		podName := "pod-" + string(uuid.NewUUID())
		volumename := fmt.Sprintf("volume%v", i+1)
		pod := makePod(namespace, podName, "local-pvc", volumename)
		_, err = testClient.CoreV1().Pods(namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
		if err != nil {
			t.Errorf("Failed to create pod: %v", err)
		}
		createPods = append(createPods, pod)
	}

	//delete all pod
	for i := 0; i < len(createPods); i++ {
		integration.DeletePodOrErrorf(t, testClient, namespace, createPods[i].Name)
	}

}

func createLocalPV(name string) *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource:        v1.PersistentVolumeSource{Local: &v1.LocalVolumeSource{Path: "/tmp"}},
			Capacity:                      v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G")},
			PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimRetain,
			AccessModes:                   []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			NodeAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "kubernetes.io/hostname",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-node"},
								},
							},
						},
					},
				},
			},
		},
	}
}

func createLocalPVC(name, namespace string) *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			Resources:   v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G")}},
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		},
	}
}

func makePod(ns, name, localPvcName, mountSubPath string) *v1.Pod {

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "fake-name",
					Image: "fakeimage",
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "local-pv",
							MountPath: fmt.Sprintf("/mnt/%s", mountSubPath),
						},
					},
				},
			},

			Volumes: []v1.Volume{
				{
					Name: "local-pv",
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: localPvcName,
							ReadOnly:  false,
						},
					},
				}},
		},
	}

	return pod
}
