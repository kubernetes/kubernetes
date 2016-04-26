/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

// If you make changes to this file, you should also make the corresponding change in ReplicationController.

package subreplicaset

import (
	"fmt"

	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface for things that know how to store SubReplicaSets.
type Registry interface {
	ListSubReplicaSets(ctx api.Context, options *api.ListOptions) (*federation.SubReplicaSetList, error)
	WatchSubReplicaSets(ctx api.Context, options *api.ListOptions) (watch.Interface, error)
	GetSubReplicaSet(ctx api.Context, subReplicaSetName string) (*federation.SubReplicaSet, error)
	CreateSubReplicaSet(ctx api.Context, subReplicaSet *federation.SubReplicaSet) (*federation.SubReplicaSet, error)
	UpdateSubReplicaSet(ctx api.Context, subReplicaSet *federation.SubReplicaSet) (*federation.SubReplicaSet, error)
	DeleteSubReplicaSet(ctx api.Context, subReplicaSetName string) error
}

// storage puts strong typing around storage calls
type storage struct {
	rest.StandardStorage
}

// NewRegistry returns a new Registry interface for the given Storage. Any mismatched
// types will panic.
func NewRegistry(s rest.StandardStorage) Registry {
	return &storage{s}
}

func (s *storage) ListSubReplicaSets(ctx api.Context, options *api.ListOptions) (*federation.SubReplicaSetList, error) {
	if options != nil && options.FieldSelector != nil && !options.FieldSelector.Empty() {
		return nil, fmt.Errorf("field selector not supported yet")
	}
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*federation.SubReplicaSetList), err
}

func (s *storage) WatchSubReplicaSets(ctx api.Context, options *api.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetSubReplicaSet(ctx api.Context, subReplicaSetName string) (*federation.SubReplicaSet, error) {
	fmt.Printf("== GetSubReplicaSet1: %v\n", subReplicaSetName)
	obj, err := s.Get(ctx, subReplicaSetName)
	fmt.Printf("== GetSubReplicaSet2: %v, err: %v\n", obj, err)
	if err != nil {
		return nil, err
	}
	return obj.(*federation.SubReplicaSet), nil
}

func (s *storage) CreateSubReplicaSet(ctx api.Context, subReplicaSet *federation.SubReplicaSet) (*federation.SubReplicaSet, error) {
	obj, err := s.Create(ctx, subReplicaSet)
	if err != nil {
		return nil, err
	}
	return obj.(*federation.SubReplicaSet), nil
}

func (s *storage) UpdateSubReplicaSet(ctx api.Context, subReplicaSet *federation.SubReplicaSet) (*federation.SubReplicaSet, error) {
	obj, _, err := s.Update(ctx, subReplicaSet)
	if err != nil {
		return nil, err
	}
	return obj.(*federation.SubReplicaSet), nil
}

func (s *storage) DeleteSubReplicaSet(ctx api.Context, subReplicaSetName string) error {
	_, err := s.Delete(ctx, subReplicaSetName, nil)
	return err
}
