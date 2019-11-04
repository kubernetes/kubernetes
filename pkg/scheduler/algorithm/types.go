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

package algorithm

import (
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	appslisters "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// NodeFieldSelectorKeys is a map that: the keys are node field selector keys; the values are
// the functions to get the value of the node field.
var NodeFieldSelectorKeys = map[string]func(*v1.Node) string{
	api.ObjectNameField: func(n *v1.Node) string { return n.Name },
}

var _ corelisters.ReplicationControllerLister = &EmptyControllerLister{}

// EmptyControllerLister implements ControllerLister on []v1.ReplicationController returning empty data
type EmptyControllerLister struct{}

// List returns nil
func (f EmptyControllerLister) List(labels.Selector) ([]*v1.ReplicationController, error) {
	return nil, nil
}

// GetPodControllers returns nil
func (f EmptyControllerLister) GetPodControllers(pod *v1.Pod) (controllers []*v1.ReplicationController, err error) {
	return nil, nil
}

// ReplicationControllers returns nil
func (f EmptyControllerLister) ReplicationControllers(namespace string) corelisters.ReplicationControllerNamespaceLister {
	return nil
}

var _ appslisters.ReplicaSetLister = &EmptyReplicaSetLister{}

// EmptyReplicaSetLister implements ReplicaSetLister on []extensions.ReplicaSet returning empty data
type EmptyReplicaSetLister struct{}

// List returns nil
func (f EmptyReplicaSetLister) List(labels.Selector) ([]*appsv1.ReplicaSet, error) {
	return nil, nil
}

// GetPodReplicaSets returns nil
func (f EmptyReplicaSetLister) GetPodReplicaSets(pod *v1.Pod) (rss []*appsv1.ReplicaSet, err error) {
	return nil, nil
}

// ReplicaSets returns nil
func (f EmptyReplicaSetLister) ReplicaSets(namespace string) appslisters.ReplicaSetNamespaceLister {
	return nil
}

// StatefulSetLister interface represents anything that can produce a list of StatefulSet; the list is consumed by a scheduler.
type StatefulSetLister interface {
	// Gets the StatefulSet for the given pod.
	GetPodStatefulSets(*v1.Pod) ([]*apps.StatefulSet, error)
}

var _ appslisters.StatefulSetLister = &EmptyStatefulSetLister{}

// EmptyStatefulSetLister implements StatefulSetLister on []apps.StatefulSet returning empty data.
type EmptyStatefulSetLister struct{}

// List returns nil
func (f EmptyStatefulSetLister) List(labels.Selector) ([]*appsv1.StatefulSet, error) {
	return nil, nil
}

// GetPodStatefulSets of EmptyStatefulSetLister returns nil.
func (f EmptyStatefulSetLister) GetPodStatefulSets(pod *v1.Pod) (sss []*appsv1.StatefulSet, err error) {
	return nil, nil
}

// StatefulSets returns nil
func (f EmptyStatefulSetLister) StatefulSets(namespace string) appslisters.StatefulSetNamespaceLister {
	return nil
}
