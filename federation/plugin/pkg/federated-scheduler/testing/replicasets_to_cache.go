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
	federation "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/schedulercache"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

// ReplicaSetsToCache is used for testing
type ReplicaSetsToCache []*extensions.ReplicaSet

func (r ReplicaSetsToCache) AssumeSubRSIfBindSucceed(subRS *federation.SubReplicaSet, bind func() bool) error {
	if !bind() {
		return nil
	}
	return nil
}

func (r ReplicaSetsToCache) AddSubRS(subRS *federation.SubReplicaSet) error { return nil }

func (r ReplicaSetsToCache) UpdateSubRS(oldSubRS, newSubRS *federation.SubReplicaSet) error { return nil }

func (r ReplicaSetsToCache) RemoveSubRS(subRS *federation.SubReplicaSet) error { return nil }

func (r ReplicaSetsToCache) GetClusterNameToInfoMap() (map[string]*schedulercache.ClusterInfo, error) {
	return nil, nil
}

func (r ReplicaSetsToCache) List() (selected []*federation.SubReplicaSet, err error) {
	return nil, nil
}
