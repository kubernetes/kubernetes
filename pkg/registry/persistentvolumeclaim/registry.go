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

package persistentvolumeclaim

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface implemented by things that know how to store PersistentVolumeClaim objects.
type Registry interface {
	// ListPersistentVolumeClaims obtains a list of PVCs having labels which match selector.
	ListPersistentVolumeClaims(ctx api.Context, selector labels.Selector) (*api.PersistentVolumeClaimList, error)
	// Watch for new/changed/deleted PVCs
	WatchPersistentVolumeClaims(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	// Get a specific PVC
	GetPersistentVolumeClaim(ctx api.Context, pvcID string) (*api.PersistentVolumeClaim, error)
	// Create a PVC based on a specification.
	CreatePersistentVolumeClaim(ctx api.Context, pvc *api.PersistentVolumeClaim) error
	// Update an existing PVC
	UpdatePersistentVolumeClaim(ctx api.Context, pvc *api.PersistentVolumeClaim) error
	// Delete an existing PVC
	DeletePersistentVolumeClaim(ctx api.Context, pvcID string) error
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

func (s *storage) ListPersistentVolumeClaims(ctx api.Context, label labels.Selector) (*api.PersistentVolumeClaimList, error) {
	obj, err := s.List(ctx, label, fields.Everything())
	if err != nil {
		return nil, err
	}
	return obj.(*api.PersistentVolumeClaimList), nil
}

func (s *storage) WatchPersistentVolumeClaims(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return s.Watch(ctx, label, field, resourceVersion)
}

func (s *storage) GetPersistentVolumeClaim(ctx api.Context, podID string) (*api.PersistentVolumeClaim, error) {
	obj, err := s.Get(ctx, podID)
	if err != nil {
		return nil, err
	}
	return obj.(*api.PersistentVolumeClaim), nil
}

func (s *storage) CreatePersistentVolumeClaim(ctx api.Context, pod *api.PersistentVolumeClaim) error {
	_, err := s.Create(ctx, pod)
	return err
}

func (s *storage) UpdatePersistentVolumeClaim(ctx api.Context, pod *api.PersistentVolumeClaim) error {
	_, _, err := s.Update(ctx, pod)
	return err
}

func (s *storage) DeletePersistentVolumeClaim(ctx api.Context, podID string) error {
	_, err := s.Delete(ctx, podID, nil)
	return err
}
