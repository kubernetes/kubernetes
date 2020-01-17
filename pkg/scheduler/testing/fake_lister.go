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

package testing

import (
	"fmt"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
)

var _ algorithm.NodeLister = &FakeNodeLister{}

// FakeNodeLister implements NodeLister on a []string for test purposes.
type FakeNodeLister []*v1.Node

// ListNodes returns nodes as a []*v1.Node.
func (f FakeNodeLister) ListNodes() []*v1.Node {
	return f
}

var _ algorithm.PodLister = &FakePodLister{}

// FakePodLister implements PodLister on an []v1.Pods for test purposes.
type FakePodLister []*v1.Pod

// List returns []*v1.Pod matching a query.
func (f FakePodLister) List(s labels.Selector) (selected []*v1.Pod, err error) {
	for _, pod := range f {
		if s.Matches(labels.Set(pod.Labels)) {
			selected = append(selected, pod)
		}
	}
	return selected, nil
}

// FilteredList returns pods matching a pod filter and a label selector.
func (f FakePodLister) FilteredList(podFilter algorithm.PodFilter, s labels.Selector) (selected []*v1.Pod, err error) {
	for _, pod := range f {
		if podFilter(pod) && s.Matches(labels.Set(pod.Labels)) {
			selected = append(selected, pod)
		}
	}
	return selected, nil
}

var _ algorithm.ServiceLister = &FakeServiceLister{}

// FakeServiceLister implements ServiceLister on []v1.Service for test purposes.
type FakeServiceLister []*v1.Service

// List returns v1.ServiceList, the list of all services.
func (f FakeServiceLister) List(labels.Selector) ([]*v1.Service, error) {
	return f, nil
}

// GetPodServices gets the services that have the selector that match the labels on the given pod.
func (f FakeServiceLister) GetPodServices(pod *v1.Pod) (services []*v1.Service, err error) {
	var selector labels.Selector

	for i := range f {
		service := f[i]
		// consider only services that are in the same namespace as the pod
		if service.Namespace != pod.Namespace {
			continue
		}
		selector = labels.Set(service.Spec.Selector).AsSelectorPreValidated()
		if selector.Matches(labels.Set(pod.Labels)) {
			services = append(services, service)
		}
	}
	return
}

var _ algorithm.ControllerLister = &FakeControllerLister{}

// FakeControllerLister implements ControllerLister on []v1.ReplicationController for test purposes.
type FakeControllerLister []*v1.ReplicationController

// List returns []v1.ReplicationController, the list of all ReplicationControllers.
func (f FakeControllerLister) List(labels.Selector) ([]*v1.ReplicationController, error) {
	return f, nil
}

// GetPodControllers gets the ReplicationControllers that have the selector that match the labels on the given pod
func (f FakeControllerLister) GetPodControllers(pod *v1.Pod) (controllers []*v1.ReplicationController, err error) {
	var selector labels.Selector

	for i := range f {
		controller := f[i]
		if controller.Namespace != pod.Namespace {
			continue
		}
		selector = labels.Set(controller.Spec.Selector).AsSelectorPreValidated()
		if selector.Matches(labels.Set(pod.Labels)) {
			controllers = append(controllers, controller)
		}
	}
	if len(controllers) == 0 {
		err = fmt.Errorf("Could not find Replication Controller for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}

	return
}

var _ algorithm.ReplicaSetLister = &FakeReplicaSetLister{}

// FakeReplicaSetLister implements ControllerLister on []extensions.ReplicaSet for test purposes.
type FakeReplicaSetLister []*apps.ReplicaSet

// GetPodReplicaSets gets the ReplicaSets that have the selector that match the labels on the given pod
func (f FakeReplicaSetLister) GetPodReplicaSets(pod *v1.Pod) (rss []*apps.ReplicaSet, err error) {
	var selector labels.Selector

	for _, rs := range f {
		if rs.Namespace != pod.Namespace {
			continue
		}
		selector, err = metav1.LabelSelectorAsSelector(rs.Spec.Selector)
		if err != nil {
			return
		}

		if selector.Matches(labels.Set(pod.Labels)) {
			rss = append(rss, rs)
		}
	}
	if len(rss) == 0 {
		err = fmt.Errorf("Could not find ReplicaSet for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}

	return
}

var _ algorithm.StatefulSetLister = &FakeStatefulSetLister{}

// FakeStatefulSetLister implements ControllerLister on []apps.StatefulSet for testing purposes.
type FakeStatefulSetLister []*apps.StatefulSet

// GetPodStatefulSets gets the StatefulSets that have the selector that match the labels on the given pod.
func (f FakeStatefulSetLister) GetPodStatefulSets(pod *v1.Pod) (sss []*apps.StatefulSet, err error) {
	var selector labels.Selector

	for _, ss := range f {
		if ss.Namespace != pod.Namespace {
			continue
		}
		selector, err = metav1.LabelSelectorAsSelector(ss.Spec.Selector)
		if err != nil {
			return
		}
		if selector.Matches(labels.Set(pod.Labels)) {
			sss = append(sss, ss)
		}
	}
	if len(sss) == 0 {
		err = fmt.Errorf("Could not find StatefulSet for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}
	return
}

// FakePersistentVolumeClaimLister implements PersistentVolumeClaimLister on []*v1.PersistentVolumeClaim for test purposes.
type FakePersistentVolumeClaimLister []*v1.PersistentVolumeClaim

var _ corelisters.PersistentVolumeClaimLister = FakePersistentVolumeClaimLister{}

// List returns not implemented error.
func (f FakePersistentVolumeClaimLister) List(selector labels.Selector) (ret []*v1.PersistentVolumeClaim, err error) {
	return nil, fmt.Errorf("not implemented")
}

// PersistentVolumeClaims returns a FakePersistentVolumeClaimLister object.
func (f FakePersistentVolumeClaimLister) PersistentVolumeClaims(namespace string) corelisters.PersistentVolumeClaimNamespaceLister {
	return &fakePersistentVolumeClaimNamespaceLister{
		pvcs:      f,
		namespace: namespace,
	}
}

// fakePersistentVolumeClaimNamespaceLister is implementation of PersistentVolumeClaimNamespaceLister returned by List() above.
type fakePersistentVolumeClaimNamespaceLister struct {
	pvcs      []*v1.PersistentVolumeClaim
	namespace string
}

func (f *fakePersistentVolumeClaimNamespaceLister) Get(name string) (*v1.PersistentVolumeClaim, error) {
	for _, pvc := range f.pvcs {
		if pvc.Name == name && pvc.Namespace == f.namespace {
			return pvc, nil
		}
	}
	return nil, fmt.Errorf("persistentvolumeclaim %q not found", name)
}

func (f fakePersistentVolumeClaimNamespaceLister) List(selector labels.Selector) (ret []*v1.PersistentVolumeClaim, err error) {
	return nil, fmt.Errorf("not implemented")
}

// FakePDBLister implements PDBLister on a slice of PodDisruptionBudgets for test purposes.
type FakePDBLister []*policy.PodDisruptionBudget

// List returns a list of PodDisruptionBudgets.
func (f FakePDBLister) List(labels.Selector) ([]*policy.PodDisruptionBudget, error) {
	return f, nil
}
