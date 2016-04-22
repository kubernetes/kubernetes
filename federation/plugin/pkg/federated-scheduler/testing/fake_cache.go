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
	federation "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
)

// FakeCache is used for testing
type FakeCache struct {
	AssumeFunc func(*federation.SubReplicaSet)
}

func (f *FakeCache) AssumeSubRSIfBindSucceed(replicaSet *federation.SubReplicaSet, bind func() bool) error {
	if !bind() {
		return nil
	}
	f.AssumeFunc(replicaSet)
	return nil
}

func (f *FakeCache) AddSubRS(replicaSet *federation.SubReplicaSet) error { return nil }

func (f *FakeCache) UpdateSubRS(oldReplicaSet, newReplicaSet *federation.SubReplicaSet) error { return nil }

func (f *FakeCache) RemoveSubRS(replicaSet *federation.SubReplicaSet) error { return nil }

func (f *FakeCache) GetClusterNameToInfoMap() (map[string]*schedulercache.ClusterInfo, error) {
	return nil, nil
}

func (f *FakeCache) List() ([]*federation.SubReplicaSet, error) { return nil, nil }
