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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/intstr"
)

// testReplenishment lets us test replenishment functions are invoked
type testReplenishment struct {
	groupKind unversioned.GroupKind
	namespace string
}

// mock function that holds onto the last kind that was replenished
func (t *testReplenishment) Replenish(groupKind unversioned.GroupKind, namespace string, object runtime.Object) {
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
	oldPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Namespace: "test", Name: "pod"},
		Status:     api.PodStatus{Phase: api.PodRunning},
	}
	newPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Namespace: "test", Name: "pod"},
		Status:     api.PodStatus{Phase: api.PodFailed},
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
	oldPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Namespace: "test", Name: "pod"},
		Status:     api.PodStatus{Phase: api.PodRunning},
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
	oldService := &api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "test", Name: "mysvc"},
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeNodePort,
			Ports: []api.ServicePort{{
				Port:       80,
				TargetPort: intstr.FromInt(80),
			}},
		},
	}
	newService := &api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "test", Name: "mysvc"},
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
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
	oldService = &api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "test", Name: "mysvc"},
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeNodePort,
			Ports: []api.ServicePort{{
				Port:       80,
				TargetPort: intstr.FromInt(80),
			}},
		},
	}
	newService = &api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "test", Name: "mysvc"},
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeNodePort,
			Ports: []api.ServicePort{{
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
