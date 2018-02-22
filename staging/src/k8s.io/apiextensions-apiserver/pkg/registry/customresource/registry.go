/*
Copyright 2018 The Kubernetes Authors.

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

package customresource

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
)

// Registry is an interface for things that know how to store CustomResources.
type Registry interface {
	ListCustomResources(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*unstructured.UnstructuredList, error)
	WatchCustomResources(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error)
	GetCustomResource(ctx genericapirequest.Context, customResourceID string, options *metav1.GetOptions) (*unstructured.Unstructured, error)
	CreateCustomResource(ctx genericapirequest.Context, customResource *unstructured.Unstructured, createValidation rest.ValidateObjectFunc) (*unstructured.Unstructured, error)
	UpdateCustomResource(ctx genericapirequest.Context, customResource *unstructured.Unstructured, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc) (*unstructured.Unstructured, error)
	DeleteCustomResource(ctx genericapirequest.Context, customResourceID string) error
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

func (s *storage) ListCustomResources(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*unstructured.UnstructuredList, error) {
	if options != nil && options.FieldSelector != nil && !options.FieldSelector.Empty() {
		return nil, fmt.Errorf("field selector not supported yet")
	}
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*unstructured.UnstructuredList), err
}

func (s *storage) WatchCustomResources(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetCustomResource(ctx genericapirequest.Context, customResourceID string, options *metav1.GetOptions) (*unstructured.Unstructured, error) {
	obj, err := s.Get(ctx, customResourceID, options)
	customResource, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return nil, fmt.Errorf("custom resource must be of type Unstructured")
	}

	if err != nil {
		apiVersion := customResource.GetAPIVersion()
		groupVersion := strings.Split(apiVersion, "/")
		group := groupVersion[0]
		return nil, errors.NewNotFound(schema.GroupResource{Group: group, Resource: "scale"}, customResourceID)
	}
	return customResource, nil
}

func (s *storage) CreateCustomResource(ctx genericapirequest.Context, customResource *unstructured.Unstructured, createValidation rest.ValidateObjectFunc) (*unstructured.Unstructured, error) {
	obj, err := s.Create(ctx, customResource, rest.ValidateAllObjectFunc, false)
	if err != nil {
		return nil, err
	}
	return obj.(*unstructured.Unstructured), nil
}

func (s *storage) UpdateCustomResource(ctx genericapirequest.Context, customResource *unstructured.Unstructured, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc) (*unstructured.Unstructured, error) {
	obj, _, err := s.Update(ctx, customResource.GetName(), rest.DefaultUpdatedObjectInfo(customResource), createValidation, updateValidation)
	if err != nil {
		return nil, err
	}
	return obj.(*unstructured.Unstructured), nil
}

func (s *storage) DeleteCustomResource(ctx genericapirequest.Context, customResourceID string) error {
	_, _, err := s.Delete(ctx, customResourceID, nil)
	return err
}
