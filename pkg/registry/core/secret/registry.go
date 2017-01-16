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

package secret

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/request"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
)

// Registry is an interface implemented by things that know how to store Secret objects.
type Registry interface {
	ListSecrets(ctx genericapirequest.Context, options *api.ListOptions) (*api.SecretList, error)
	WatchSecrets(ctx genericapirequest.Context, options *api.ListOptions) (watch.Interface, error)
	GetSecret(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*api.Secret, error)
	CreateSecret(ctx genericapirequest.Context, Secret *api.Secret) (*api.Secret, error)
	UpdateSecret(ctx genericapirequest.Context, Secret *api.Secret) (*api.Secret, error)
	DeleteSecret(ctx genericapirequest.Context, name string) error
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

func (s *storage) ListSecrets(ctx genericapirequest.Context, options *api.ListOptions) (*api.SecretList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*api.SecretList), nil
}

func (s *storage) WatchSecrets(ctx genericapirequest.Context, options *api.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetSecret(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*api.Secret, error) {
	obj, err := s.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	return obj.(*api.Secret), nil
}

func (s *storage) CreateSecret(ctx genericapirequest.Context, secret *api.Secret) (*api.Secret, error) {
	obj, err := s.Create(ctx, secret)
	return obj.(*api.Secret), err
}

func (s *storage) UpdateSecret(ctx genericapirequest.Context, secret *api.Secret) (*api.Secret, error) {
	obj, _, err := s.Update(ctx, secret.Name, rest.DefaultUpdatedObjectInfo(secret, api.Scheme))
	return obj.(*api.Secret), err
}

func (s *storage) DeleteSecret(ctx genericapirequest.Context, name string) error {
	_, err := s.Delete(ctx, name, nil)
	return err
}
