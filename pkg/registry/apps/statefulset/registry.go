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
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
)

// Registry is an interface for things that know how to store StatefulSets.
type Registry interface {
	ListStatefulSets(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*apps.StatefulSetList, error)
	WatchStatefulSets(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error)
	GetStatefulSet(ctx genericapirequest.Context, statefulSetID string, options *metav1.GetOptions) (*apps.StatefulSet, error)
	CreateStatefulSet(ctx genericapirequest.Context, statefulSet *apps.StatefulSet) (*apps.StatefulSet, error)
	UpdateStatefulSet(ctx genericapirequest.Context, statefulSet *apps.StatefulSet) (*apps.StatefulSet, error)
	DeleteStatefulSet(ctx genericapirequest.Context, statefulSetID string) error
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

func (s *storage) ListStatefulSets(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*apps.StatefulSetList, error) {
	if options != nil && options.FieldSelector != nil && !options.FieldSelector.Empty() {
		return nil, fmt.Errorf("field selector not supported yet")
	}
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*apps.StatefulSetList), err
}

func (s *storage) WatchStatefulSets(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetStatefulSet(ctx genericapirequest.Context, statefulSetID string, options *metav1.GetOptions) (*apps.StatefulSet, error) {
	obj, err := s.Get(ctx, statefulSetID, options)
	if err != nil {
		return nil, errors.NewNotFound(apps.Resource("statefulsets/scale"), statefulSetID)
	}
	return obj.(*apps.StatefulSet), nil
}

func (s *storage) CreateStatefulSet(ctx genericapirequest.Context, statefulSet *apps.StatefulSet) (*apps.StatefulSet, error) {
	obj, err := s.Create(ctx, statefulSet, false)
	if err != nil {
		return nil, err
	}
	return obj.(*apps.StatefulSet), nil
}

func (s *storage) UpdateStatefulSet(ctx genericapirequest.Context, statefulSet *apps.StatefulSet) (*apps.StatefulSet, error) {
	obj, _, err := s.Update(ctx, statefulSet.Name, rest.DefaultUpdatedObjectInfo(statefulSet, api.Scheme))
	if err != nil {
		return nil, err
	}
	return obj.(*apps.StatefulSet), nil
}

func (s *storage) DeleteStatefulSet(ctx genericapirequest.Context, statefulSetID string) error {
	_, _, err := s.Delete(ctx, statefulSetID, nil)
	return err
}
