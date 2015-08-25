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

package horizontalpodautoscaler

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

// Registry is an interface implemented by things that know how to store HorizontalPodAutoscaler objects.
type Registry interface {
	// ListPersistentVolumes obtains a list of autoscalers having labels which match selector.
	ListHorizontalPodAutoscaler(ctx api.Context, selector labels.Selector) (*expapi.HorizontalPodAutoscalerList, error)
	// Get a specific autoscaler
	GetHorizontalPodAutoscaler(ctx api.Context, autoscalerID string) (*expapi.HorizontalPodAutoscaler, error)
	// Create an autoscaler based on a specification.
	CreateHorizontalPodAutoscaler(ctx api.Context, autoscaler *expapi.HorizontalPodAutoscaler) error
	// Update an existing autoscaler
	UpdateHorizontalPodAutoscaler(ctx api.Context, autoscaler *expapi.HorizontalPodAutoscaler) error
	// Delete an existing autoscaler
	DeleteHorizontalPodAutoscaler(ctx api.Context, autoscalerID string) error
}

// storage puts strong typing around storage calls
type storage struct {
	rest.StandardStorage
}

// NewREST returns a new Registry interface for the given Storage. Any mismatched types will panic.
func NewRegistry(s rest.StandardStorage) Registry {
	return &storage{s}
}

func (s *storage) ListHorizontalPodAutoscaler(ctx api.Context, label labels.Selector) (*expapi.HorizontalPodAutoscalerList, error) {
	obj, err := s.List(ctx, label, fields.Everything())
	if err != nil {
		return nil, err
	}
	return obj.(*expapi.HorizontalPodAutoscalerList), nil
}

func (s *storage) GetHorizontalPodAutoscaler(ctx api.Context, autoscalerID string) (*expapi.HorizontalPodAutoscaler, error) {
	obj, err := s.Get(ctx, autoscalerID)
	if err != nil {
		return nil, err
	}
	return obj.(*expapi.HorizontalPodAutoscaler), nil
}

func (s *storage) CreateHorizontalPodAutoscaler(ctx api.Context, autoscaler *expapi.HorizontalPodAutoscaler) error {
	_, err := s.Create(ctx, autoscaler)
	return err
}

func (s *storage) UpdateHorizontalPodAutoscaler(ctx api.Context, autoscaler *expapi.HorizontalPodAutoscaler) error {
	_, _, err := s.Update(ctx, autoscaler)
	return err
}

func (s *storage) DeleteHorizontalPodAutoscaler(ctx api.Context, autoscalerID string) error {
	_, err := s.Delete(ctx, autoscalerID, nil)
	return err
}
