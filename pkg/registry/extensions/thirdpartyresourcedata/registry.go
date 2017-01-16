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

package thirdpartyresourcedata

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/request"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

// Registry is an interface implemented by things that know how to store ThirdPartyResourceData objects.
type Registry interface {
	ListThirdPartyResourceData(ctx genericapirequest.Context, options *api.ListOptions) (*extensions.ThirdPartyResourceDataList, error)
	WatchThirdPartyResourceData(ctx genericapirequest.Context, options *api.ListOptions) (watch.Interface, error)
	GetThirdPartyResourceData(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*extensions.ThirdPartyResourceData, error)
	CreateThirdPartyResourceData(ctx genericapirequest.Context, resource *extensions.ThirdPartyResourceData) (*extensions.ThirdPartyResourceData, error)
	UpdateThirdPartyResourceData(ctx genericapirequest.Context, resource *extensions.ThirdPartyResourceData) (*extensions.ThirdPartyResourceData, error)
	DeleteThirdPartyResourceData(ctx genericapirequest.Context, name string) error
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

func (s *storage) ListThirdPartyResourceData(ctx genericapirequest.Context, options *api.ListOptions) (*extensions.ThirdPartyResourceDataList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*extensions.ThirdPartyResourceDataList), nil
}

func (s *storage) WatchThirdPartyResourceData(ctx genericapirequest.Context, options *api.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetThirdPartyResourceData(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*extensions.ThirdPartyResourceData, error) {
	obj, err := s.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	return obj.(*extensions.ThirdPartyResourceData), nil
}

func (s *storage) CreateThirdPartyResourceData(ctx genericapirequest.Context, ThirdPartyResourceData *extensions.ThirdPartyResourceData) (*extensions.ThirdPartyResourceData, error) {
	obj, err := s.Create(ctx, ThirdPartyResourceData)
	return obj.(*extensions.ThirdPartyResourceData), err
}

func (s *storage) UpdateThirdPartyResourceData(ctx genericapirequest.Context, ThirdPartyResourceData *extensions.ThirdPartyResourceData) (*extensions.ThirdPartyResourceData, error) {
	obj, _, err := s.Update(ctx, ThirdPartyResourceData.Name, rest.DefaultUpdatedObjectInfo(ThirdPartyResourceData, api.Scheme))
	return obj.(*extensions.ThirdPartyResourceData), err
}

func (s *storage) DeleteThirdPartyResourceData(ctx genericapirequest.Context, name string) error {
	_, err := s.Delete(ctx, name, nil)
	return err
}
