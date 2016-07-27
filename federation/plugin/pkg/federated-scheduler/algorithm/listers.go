/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	federation "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

var _ ReplicaSetLister = &EmptyReplicaSetLister{}

//ClusterLister interface represents anything that can list clusters for a federated-scheduler.
type ClusterLister interface {
	List() (list federation.ClusterList, err error)
}

// FakeClusterLister implements ClusterLister on a []string for test purposes.
type FakeClusterLister federation.ClusterList

// List returns clusters as a []string.
func (f FakeClusterLister) List() (federation.ClusterList, error) {
	return federation.ClusterList(f), nil
}

// FakeControllerLister implements ControllerLister on []extensions.ReplicaSet for test purposes.
type FakeControllerLister []extensions.ReplicaSet

// List returns []extensions.ReplicaSet, the list of all ReplicationControllers.
func (f FakeControllerLister) List() ([]extensions.ReplicaSet, error) {
	return f, nil
}

// ReplicaSetLister interface represents anything that can produce a list of ReplicaSet; the list is consumed by a federated-scheduler.
type ReplicaSetLister interface {
	// Lists all the replicasets
	List() ([]extensions.ReplicaSet, error)
}

// EmptyReplicaSetLister implements ReplicaSetLister on []extensions.ReplicaSet returning empty data
type EmptyReplicaSetLister struct{}

// List returns nil
func (f EmptyReplicaSetLister) List() ([]extensions.ReplicaSet, error) {
	return nil, nil
}

// FakeReplicaSetLister implements ControllerLister on []extensions.ReplicaSet for test purposes.
type FakeReplicaSetLister []extensions.ReplicaSet

// List returns []extensions.ReplicaSet, the list of all ReplicaSets.
func (f FakeReplicaSetLister) List() ([]extensions.ReplicaSet, error) {
	return f, nil
}
