/*
Copyright 2014 The Kubernetes Authors.

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

package serviceaccount

import (
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
)

// Registry is an interface implemented by things that know how to store ServiceAccount objects.
type Registry interface {
	ListServiceAccounts(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*api.ServiceAccountList, error)
	WatchServiceAccounts(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error)
	GetServiceAccount(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*api.ServiceAccount, error)
	CreateServiceAccount(ctx genericapirequest.Context, ServiceAccount *api.ServiceAccount) error
	UpdateServiceAccount(ctx genericapirequest.Context, ServiceAccount *api.ServiceAccount) error
	DeleteServiceAccount(ctx genericapirequest.Context, name string) error
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

func (s *storage) ListServiceAccounts(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*api.ServiceAccountList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*api.ServiceAccountList), nil
}

func (s *storage) WatchServiceAccounts(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetServiceAccount(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*api.ServiceAccount, error) {
	obj, err := s.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	return obj.(*api.ServiceAccount), nil
}

func (s *storage) CreateServiceAccount(ctx genericapirequest.Context, serviceAccount *api.ServiceAccount) error {
	_, err := s.Create(ctx, serviceAccount)
	return err
}

func (s *storage) UpdateServiceAccount(ctx genericapirequest.Context, serviceAccount *api.ServiceAccount) error {
	_, _, err := s.Update(ctx, serviceAccount.Name, rest.DefaultUpdatedObjectInfo(serviceAccount, api.Scheme))
	return err
}

func (s *storage) DeleteServiceAccount(ctx genericapirequest.Context, name string) error {
	_, _, err := s.Delete(ctx, name, nil)
	return err
}
