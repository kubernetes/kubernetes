/*
Copyright 2019 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/labels"
)

var _ ControllerLister = &EmptyControllerLister{}

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

var _ ReplicaSetLister = &EmptyReplicaSetLister{}

// EmptyReplicaSetLister implements ReplicaSetLister on []extensions.ReplicaSet returning empty data
type EmptyReplicaSetLister struct{}

// GetPodReplicaSets returns nil
func (f EmptyReplicaSetLister) GetPodReplicaSets(pod *v1.Pod) (rss []*apps.ReplicaSet, err error) {
	return nil, nil
}

var _ StatefulSetLister = &EmptyStatefulSetLister{}

// EmptyStatefulSetLister implements StatefulSetLister on []apps.StatefulSet returning empty data.
type EmptyStatefulSetLister struct{}

// GetPodStatefulSets of EmptyStatefulSetLister returns nil.
func (f EmptyStatefulSetLister) GetPodStatefulSets(pod *v1.Pod) (sss []*apps.StatefulSet, err error) {
	return nil, nil
}
