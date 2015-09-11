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

package lock

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface implemented by things that know how to store Lock objects.
type Registry interface {
	// ListLocks obtains a list of locks having labels which match selector.
	ListLocks(ctx api.Context, selector labels.Selector) (*expapi.LockList, error)
	// Watch for new/changed/deleted locks
	WatchLocks(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	// Get a specific lock
	GetLock(ctx api.Context, lockName string) (*expapi.Lock, error)
	// Create a lock based on a specification.
	CreateLock(ctx api.Context, lock *expapi.Lock) error
	// Update an existing lock
	UpdateLock(ctx api.Context, lock *expapi.Lock) error
	// Delete an existing lock
	DeleteLock(ctx api.Context, lockName string) error
}

type storage struct {
	rest.StandardStorage
}

// NewRegistry returns a new Registry interface for the given Storage. Any mismatched
// types will panic.
func NewRegistry(s rest.StandardStorage) Registry {
	return &storage{s}
}

func (s *storage) ListLocks(ctx api.Context, label labels.Selector) (*expapi.LockList, error) {
	obj, err := s.List(ctx, label, fields.Everything())
	if err != nil {
		return nil, err
	}
	return obj.(*expapi.LockList), nil
}

func (s *storage) WatchLocks(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return s.Watch(ctx, label, field, resourceVersion)
}

func (s *storage) GetLock(ctx api.Context, lockName string) (*expapi.Lock, error) {
	obj, err := s.Get(ctx, lockName)
	if err != nil {
		return nil, err
	}
	return obj.(*expapi.Lock), nil
}

func (s *storage) CreateLock(ctx api.Context, lockName *expapi.Lock) error {
	_, err := s.Create(ctx, lockName)
	return err
}

func (s *storage) UpdateLock(ctx api.Context, lockName *expapi.Lock) error {
	_, _, err := s.Update(ctx, lockName)
	return err
}

func (s *storage) DeleteLock(ctx api.Context, lockName string) error {
	_, err := s.Delete(ctx, lockName, nil)
	return err
}
