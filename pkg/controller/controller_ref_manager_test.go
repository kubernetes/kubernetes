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

package controller

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
)

var (
	productionLabel         = map[string]string{"type": "production"}
	testLabel               = map[string]string{"type": "testing"}
	productionLabelSelector = labels.Set{"type": "production"}.AsSelector()
	testLabelSelector       = labels.Set{"type": "testing"}.AsSelector()
	controllerUID           = "123"
)

func newPod(podName string, label map[string]string, owner metav1.Object) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Labels:    label,
			Namespace: metav1.NamespaceDefault,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Image: "foo/bar",
				},
			},
		},
	}
	if owner != nil {
		pod.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(owner, v1beta1.SchemeGroupVersion.WithKind("Fake"))}
	}
	return pod
}

func TestClaimPods(t *testing.T) {
	controllerKind := schema.GroupVersionKind{}
	type test struct {
		name     string
		manager  *PodControllerRefManager
		pods     []*v1.Pod
		filters  []func(*v1.Pod) bool
		claimed  []*v1.Pod
		released []*v1.Pod
	}
	var tests = []test{
		{
			name: "Claim pods with correct label",
			manager: NewPodControllerRefManager(&FakePodControl{},
				&v1.ReplicationController{},
				productionLabelSelector,
				controllerKind,
				func() error { return nil }),
			pods:    []*v1.Pod{newPod("pod1", productionLabel, nil), newPod("pod2", testLabel, nil)},
			claimed: []*v1.Pod{newPod("pod1", productionLabel, nil)},
		},
		func() test {
			controller := v1.ReplicationController{}
			controller.UID = types.UID(controllerUID)
			now := metav1.Now()
			controller.DeletionTimestamp = &now
			return test{
				name: "Controller marked for deletion can not claim pods",
				manager: NewPodControllerRefManager(&FakePodControl{},
					&controller,
					productionLabelSelector,
					controllerKind,
					func() error { return nil }),
				pods:    []*v1.Pod{newPod("pod1", productionLabel, nil), newPod("pod2", productionLabel, nil)},
				claimed: nil,
			}
		}(),
		func() test {
			controller := v1.ReplicationController{}
			controller.UID = types.UID(controllerUID)
			now := metav1.Now()
			controller.DeletionTimestamp = &now
			return test{
				name: "Controller marked for deletion can not claim new pods",
				manager: NewPodControllerRefManager(&FakePodControl{},
					&controller,
					productionLabelSelector,
					controllerKind,
					func() error { return nil }),
				pods:    []*v1.Pod{newPod("pod1", productionLabel, &controller), newPod("pod2", productionLabel, nil)},
				claimed: []*v1.Pod{newPod("pod1", productionLabel, &controller)},
			}
		}(),
		func() test {
			controller := v1.ReplicationController{}
			controller2 := v1.ReplicationController{}
			controller.UID = types.UID(controllerUID)
			controller2.UID = types.UID("AAAAA")
			return test{
				name: "Controller can not claim pods owned by another controller",
				manager: NewPodControllerRefManager(&FakePodControl{},
					&controller,
					productionLabelSelector,
					controllerKind,
					func() error { return nil }),
				pods:    []*v1.Pod{newPod("pod1", productionLabel, &controller), newPod("pod2", productionLabel, &controller2)},
				claimed: []*v1.Pod{newPod("pod1", productionLabel, &controller)},
			}
		}(),
		func() test {
			controller := v1.ReplicationController{}
			controller.UID = types.UID(controllerUID)
			return test{
				name: "Controller releases claimed pods when selector doesn't match",
				manager: NewPodControllerRefManager(&FakePodControl{},
					&controller,
					productionLabelSelector,
					controllerKind,
					func() error { return nil }),
				pods:    []*v1.Pod{newPod("pod1", productionLabel, &controller), newPod("pod2", testLabel, &controller)},
				claimed: []*v1.Pod{newPod("pod1", productionLabel, &controller)},
			}
		}(),
		func() test {
			controller := v1.ReplicationController{}
			controller.UID = types.UID(controllerUID)
			podToDelete1 := newPod("pod1", productionLabel, &controller)
			podToDelete2 := newPod("pod2", productionLabel, nil)
			now := metav1.Now()
			podToDelete1.DeletionTimestamp = &now
			podToDelete2.DeletionTimestamp = &now

			return test{
				name: "Controller does not claim orphaned pods marked for deletion",
				manager: NewPodControllerRefManager(&FakePodControl{},
					&controller,
					productionLabelSelector,
					controllerKind,
					func() error { return nil }),
				pods:    []*v1.Pod{podToDelete1, podToDelete2},
				claimed: []*v1.Pod{podToDelete1},
			}
		}(),
	}
	for _, test := range tests {
		claimed, err := test.manager.ClaimPods(test.pods)
		if err != nil {
			t.Errorf("Test case `%s`, unexpected error: %v", test.name, err)
		} else if !reflect.DeepEqual(test.claimed, claimed) {
			t.Errorf("Test case `%s`, claimed wrong pods. Expected %v, got %v", test.name, podToStringSlice(test.claimed), podToStringSlice(claimed))
		}

	}
}

func podToStringSlice(pods []*v1.Pod) []string {
	var names []string
	for _, pod := range pods {
		names = append(names, pod.Name)
	}
	return names
}
