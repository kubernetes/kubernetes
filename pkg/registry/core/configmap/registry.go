/*
Copyright 2015 The Kubernetes Authors.

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

package configmap

import (
	"context"

	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/registry/rest"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// Registry is an interface for things that know how to store ConfigMaps.
type Registry interface {
	ListConfigMaps(ctx context.Context, options *metainternalversion.ListOptions) (*api.ConfigMapList, error)
	WatchConfigMaps(ctx context.Context, options *metainternalversion.ListOptions) (watch.Interface, error)
	GetConfigMap(ctx context.Context, name string, options *metav1.GetOptions) (*api.ConfigMap, error)
	CreateConfigMap(ctx context.Context, cfg *api.ConfigMap, createValidation rest.ValidateObjectFunc) (*api.ConfigMap, error)
	UpdateConfigMap(ctx context.Context, cfg *api.ConfigMap, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc) (*api.ConfigMap, error)
	DeleteConfigMap(ctx context.Context, name string) error
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

func (s *storage) ListConfigMaps(ctx context.Context, options *metainternalversion.ListOptions) (*api.ConfigMapList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}

	return obj.(*api.ConfigMapList), err
}

func (s *storage) WatchConfigMaps(ctx context.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetConfigMap(ctx context.Context, name string, options *metav1.GetOptions) (*api.ConfigMap, error) {
	obj, err := s.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}

	return obj.(*api.ConfigMap), nil
}

func (s *storage) CreateConfigMap(ctx context.Context, cfg *api.ConfigMap, createValidation rest.ValidateObjectFunc) (*api.ConfigMap, error) {
	obj, err := s.Create(ctx, cfg, createValidation, false)
	if err != nil {
		return nil, err
	}

	return obj.(*api.ConfigMap), nil
}

func (s *storage) UpdateConfigMap(ctx context.Context, cfg *api.ConfigMap, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc) (*api.ConfigMap, error) {
	obj, _, err := s.Update(ctx, cfg.Name, rest.DefaultUpdatedObjectInfo(cfg), createValidation, updateValidation)
	if err != nil {
		return nil, err
	}

	return obj.(*api.ConfigMap), nil
}

func (s *storage) DeleteConfigMap(ctx context.Context, name string) error {
	_, _, err := s.Delete(ctx, name, nil)
	return err
}
