/*
Copyright 2017 The Kubernetes Authors.

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

package statefulset

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/apps"
)

// Registry is an interface for things that know how to store StatefulSets.
type Registry interface {
	ListStatefulSets(ctx context.Context, options *metainternalversion.ListOptions) (*apps.StatefulSetList, error)
	WatchStatefulSets(ctx context.Context, options *metainternalversion.ListOptions) (watch.Interface, error)
	GetStatefulSet(ctx context.Context, statefulSetID string, options *metav1.GetOptions) (*apps.StatefulSet, error)
	CreateStatefulSet(ctx context.Context, statefulSet *apps.StatefulSet, createValidation rest.ValidateObjectFunc) (*apps.StatefulSet, error)
	UpdateStatefulSet(ctx context.Context, statefulSet *apps.StatefulSet, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc) (*apps.StatefulSet, error)
	DeleteStatefulSet(ctx context.Context, statefulSetID string) error
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

func (s *storage) ListStatefulSets(ctx context.Context, options *metainternalversion.ListOptions) (*apps.StatefulSetList, error) {
	if options != nil && options.FieldSelector != nil && !options.FieldSelector.Empty() {
		return nil, fmt.Errorf("field selector not supported yet")
	}
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*apps.StatefulSetList), err
}

func (s *storage) WatchStatefulSets(ctx context.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetStatefulSet(ctx context.Context, statefulSetID string, options *metav1.GetOptions) (*apps.StatefulSet, error) {
	obj, err := s.Get(ctx, statefulSetID, options)
	if err != nil {
		return nil, errors.NewNotFound(apps.Resource("statefulsets/scale"), statefulSetID)
	}
	return obj.(*apps.StatefulSet), nil
}

func (s *storage) CreateStatefulSet(ctx context.Context, statefulSet *apps.StatefulSet, createValidation rest.ValidateObjectFunc) (*apps.StatefulSet, error) {
	obj, err := s.Create(ctx, statefulSet, rest.ValidateAllObjectFunc, false)
	if err != nil {
		return nil, err
	}
	return obj.(*apps.StatefulSet), nil
}

func (s *storage) UpdateStatefulSet(ctx context.Context, statefulSet *apps.StatefulSet, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc) (*apps.StatefulSet, error) {
	obj, _, err := s.Update(ctx, statefulSet.Name, rest.DefaultUpdatedObjectInfo(statefulSet), createValidation, updateValidation)
	if err != nil {
		return nil, err
	}
	return obj.(*apps.StatefulSet), nil
}

func (s *storage) DeleteStatefulSet(ctx context.Context, statefulSetID string) error {
	_, _, err := s.Delete(ctx, statefulSetID, nil)
	return err
}
