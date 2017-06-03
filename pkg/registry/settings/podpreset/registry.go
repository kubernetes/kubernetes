/*
Copyright 2016 The Kubernetes Authors.

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

package podpreset

import (
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/settings"
)

// Registry is an interface for things that know how to store PodPresets.
type Registry interface {
	ListPodPresets(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*settings.PodPresetList, error)
	CreatePodPreset(ctx genericapirequest.Context, pp *settings.PodPreset) error
	UpdatePodPreset(ctx genericapirequest.Context, pp *settings.PodPreset) error
	GetPodPreset(ctx genericapirequest.Context, ppID string, options *metav1.GetOptions) (*settings.PodPreset, error)
	DeletePodPreset(ctx genericapirequest.Context, ppID string) error
	WatchPodPresets(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error)
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

func (s *storage) ListPodPresets(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*settings.PodPresetList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}

	return obj.(*settings.PodPresetList), nil
}

func (s *storage) CreatePodPreset(ctx genericapirequest.Context, pp *settings.PodPreset) error {
	_, err := s.Create(ctx, pp, false)
	return err
}

func (s *storage) UpdatePodPreset(ctx genericapirequest.Context, pp *settings.PodPreset) error {
	_, _, err := s.Update(ctx, pp.Name, rest.DefaultUpdatedObjectInfo(pp, api.Scheme))
	return err
}

func (s *storage) WatchPodPresets(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetPodPreset(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*settings.PodPreset, error) {
	obj, err := s.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	return obj.(*settings.PodPreset), nil
}

func (s *storage) DeletePodPreset(ctx genericapirequest.Context, name string) error {
	_, _, err := s.Delete(ctx, name, nil)
	return err
}
