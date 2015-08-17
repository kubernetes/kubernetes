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

package persistentvolume

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface implemented by things that know how to store PersistentVolume objects.
type Registry interface {
	// ListPersistentVolumes obtains a list of persistentVolumes having labels which match selector.
	ListPersistentVolumes(ctx api.Context, selector labels.Selector) (*api.PersistentVolumeList, error)
	// Watch for new/changed/deleted persistentVolumes
	WatchPersistentVolumes(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	// Get a specific persistentVolume
	GetPersistentVolume(ctx api.Context, persistentVolumeID string) (*api.PersistentVolume, error)
	// Create a persistentVolume based on a specification.
	CreatePersistentVolume(ctx api.Context, persistentVolume *api.PersistentVolume) error
	// Update an existing persistentVolume
	UpdatePersistentVolume(ctx api.Context, persistentVolume *api.PersistentVolume) error
	// Delete an existing persistentVolume
	DeletePersistentVolume(ctx api.Context, persistentVolumeID string) error
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

func (s *storage) ListPersistentVolumes(ctx api.Context, label labels.Selector) (*api.PersistentVolumeList, error) {
	obj, err := s.List(ctx, label, fields.Everything())
	if err != nil {
		return nil, err
	}
	return obj.(*api.PersistentVolumeList), nil
}

func (s *storage) WatchPersistentVolumes(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return s.Watch(ctx, label, field, resourceVersion)
}

func (s *storage) GetPersistentVolume(ctx api.Context, persistentVolumeID string) (*api.PersistentVolume, error) {
	obj, err := s.Get(ctx, persistentVolumeID)
	if err != nil {
		return nil, err
	}
	return obj.(*api.PersistentVolume), nil
}

func (s *storage) CreatePersistentVolume(ctx api.Context, persistentVolume *api.PersistentVolume) error {
	_, err := s.Create(ctx, persistentVolume)
	return err
}

func (s *storage) UpdatePersistentVolume(ctx api.Context, persistentVolume *api.PersistentVolume) error {
	_, _, err := s.Update(ctx, persistentVolume)
	return err
}

func (s *storage) DeletePersistentVolume(ctx api.Context, persistentVolumeID string) error {
	_, err := s.Delete(ctx, persistentVolumeID, nil)
	return err
}
