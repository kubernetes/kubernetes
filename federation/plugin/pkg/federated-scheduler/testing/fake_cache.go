/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package schedulercache

import (
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/schedulercache"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

// FakeCache is used for testing
type FakeCache struct {
	AssumeFunc func(*v1beta1.ReplicaSet)
}

func (f *FakeCache) AssumeReplicaSet(replicaSet *v1beta1.ReplicaSet) error {
	f.AssumeFunc(replicaSet)
	return nil
}

func (f *FakeCache) AddReplicaSet(replicaSet *v1beta1.ReplicaSet) error { return nil }

func (f *FakeCache) UpdateReplicaSet(oldReplicaSet, newReplicaSet *v1beta1.ReplicaSet) error { return nil }

func (f *FakeCache) RemoveReplicaSet(replicaSet *v1beta1.ReplicaSet) error { return nil }

func (f *FakeCache) GetClusterNameToInfoMap() (map[string]*schedulercache.ClusterInfo, error) {
	return nil, nil
}

func (f *FakeCache) List() ([]*v1beta1.ReplicaSet, error) { return nil, nil }
