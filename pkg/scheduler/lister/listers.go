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

package lister

import (
	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/labels"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
)

// NodeFieldSelectorKeys is a map that: the keys are node field selector keys; the values are
// the functions to get the value of the node field.
var NodeFieldSelectorKeys = map[string]func(*v1.Node) string{
	schedulerapi.NodeFieldSelectorKeyNodeName: func(n *v1.Node) string { return n.Name },
}

// PodFilter is a function to filter a pod. If pod passed return true else return false.
type PodFilter func(*v1.Pod) bool

// PodLister interface represents anything that can list pods for a scheduler.
type PodLister interface {
	// We explicitly return []*v1.Pod, instead of v1.PodList, to avoid
	// performing expensive copies that are unneeded.
	List(labels.Selector) ([]*v1.Pod, error)
	// This is similar to "List()", but the returned slice does not
	// contain pods that don't pass `podFilter`.
	FilteredList(podFilter PodFilter, selector labels.Selector) ([]*v1.Pod, error)
}

// ServiceLister interface represents anything that can produce a list of services; the list is consumed by a scheduler.
type ServiceLister interface {
	// Lists all the services
	List(labels.Selector) ([]*v1.Service, error)
	// Gets the services for the given pod
	GetPodServices(*v1.Pod) ([]*v1.Service, error)
}

// ControllerLister interface represents anything that can produce a list of ReplicationController; the list is consumed by a scheduler.
type ControllerLister interface {
	// Lists all the replication controllers
	List(labels.Selector) ([]*v1.ReplicationController, error)
	// Gets the replication controller for the given pod
	GetPodControllers(*v1.Pod) ([]*v1.ReplicationController, error)
}

// ReplicaSetLister interface represents anything that can produce a list of ReplicaSet; the list is consumed by a scheduler.
type ReplicaSetLister interface {
	// Gets the replicasets for the given pod
	GetPodReplicaSets(*v1.Pod) ([]*apps.ReplicaSet, error)
}

// PDBLister interface represents anything that can list PodDisruptionBudget objects.
type PDBLister interface {
	// List all the PodDisruptionBudgets matching the selector
	List(labels.Selector) ([]*policyv1beta1.PodDisruptionBudget, error)
}

// StatefulSetLister interface represents anything that can produce a list of StatefulSet; the list is consumed by a scheduler.
type StatefulSetLister interface {
	// Gets the StatefulSet for the given pod.
	GetPodStatefulSets(*v1.Pod) ([]*apps.StatefulSet, error)
}
