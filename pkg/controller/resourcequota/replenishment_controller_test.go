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

package resourcequota

import (
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/controller"
)

// testReplenishment lets us test replenishment functions are invoked
type testReplenishment struct {
	groupKind schema.GroupKind
	namespace string
}

// mock function that holds onto the last kind that was replenished
func (t *testReplenishment) Replenish(groupKind schema.GroupKind, namespace string, object runtime.Object) {
	t.groupKind = groupKind
	t.namespace = namespace
}

func TestPodReplenishmentUpdateFunc(t *testing.T) {
	mockReplenish := &testReplenishment{}
	options := ReplenishmentControllerOptions{
		GroupKind:         api.Kind("Pod"),
		ReplenishmentFunc: mockReplenish.Replenish,
		ResyncPeriod:      controller.NoResyncPeriodFunc,
	}
	oldPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "pod"},
		Status:     v1.PodStatus{Phase: v1.PodRunning},
	}
	newPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "pod"},
		Status:     v1.PodStatus{Phase: v1.PodFailed},
	}
	updateFunc := PodReplenishmentUpdateFunc(&options)
	updateFunc(oldPod, newPod)
	if mockReplenish.groupKind != api.Kind("Pod") {
		t.Errorf("Unexpected group kind %v", mockReplenish.groupKind)
	}
	if mockReplenish.namespace != oldPod.Namespace {
		t.Errorf("Unexpected namespace %v", mockReplenish.namespace)
	}
}

func TestObjectReplenishmentDeleteFunc(t *testing.T) {
	mockReplenish := &testReplenishment{}
	options := ReplenishmentControllerOptions{
		GroupKind:         api.Kind("Pod"),
		ReplenishmentFunc: mockReplenish.Replenish,
		ResyncPeriod:      controller.NoResyncPeriodFunc,
	}
	oldPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "pod"},
		Status:     v1.PodStatus{Phase: v1.PodRunning},
	}
	deleteFunc := ObjectReplenishmentDeleteFunc(&options)
	deleteFunc(oldPod)
	if mockReplenish.groupKind != api.Kind("Pod") {
		t.Errorf("Unexpected group kind %v", mockReplenish.groupKind)
	}
	if mockReplenish.namespace != oldPod.Namespace {
		t.Errorf("Unexpected namespace %v", mockReplenish.namespace)
	}
}

func TestServiceReplenishmentUpdateFunc(t *testing.T) {
	mockReplenish := &testReplenishment{}
	options := ReplenishmentControllerOptions{
		GroupKind:         api.Kind("Service"),
		ReplenishmentFunc: mockReplenish.Replenish,
		ResyncPeriod:      controller.NoResyncPeriodFunc,
	}
	oldService := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "mysvc"},
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeNodePort,
			Ports: []v1.ServicePort{{
				Port:       80,
				TargetPort: intstr.FromInt(80),
			}},
		},
	}
	newService := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "mysvc"},
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeClusterIP,
			Ports: []v1.ServicePort{{
				Port:       80,
				TargetPort: intstr.FromInt(80),
			}}},
	}
	updateFunc := ServiceReplenishmentUpdateFunc(&options)
	updateFunc(oldService, newService)
	if mockReplenish.groupKind != api.Kind("Service") {
		t.Errorf("Unexpected group kind %v", mockReplenish.groupKind)
	}
	if mockReplenish.namespace != oldService.Namespace {
		t.Errorf("Unexpected namespace %v", mockReplenish.namespace)
	}

	mockReplenish = &testReplenishment{}
	options = ReplenishmentControllerOptions{
		GroupKind:         api.Kind("Service"),
		ReplenishmentFunc: mockReplenish.Replenish,
		ResyncPeriod:      controller.NoResyncPeriodFunc,
	}
	oldService = &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "mysvc"},
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeNodePort,
			Ports: []v1.ServicePort{{
				Port:       80,
				TargetPort: intstr.FromInt(80),
			}},
		},
	}
	newService = &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "mysvc"},
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeNodePort,
			Ports: []v1.ServicePort{{
				Port:       81,
				TargetPort: intstr.FromInt(81),
			}}},
	}
	updateFunc = ServiceReplenishmentUpdateFunc(&options)
	updateFunc(oldService, newService)
	if mockReplenish.groupKind == api.Kind("Service") {
		t.Errorf("Unexpected group kind %v", mockReplenish.groupKind)
	}
	if mockReplenish.namespace == oldService.Namespace {
		t.Errorf("Unexpected namespace %v", mockReplenish.namespace)
	}
}
