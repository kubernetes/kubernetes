/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/secret"
	secretetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/secret/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/serviceaccount"
	serviceaccountetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/serviceaccount/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/storage"
)

// ServiceAccountTokenGetter defines functions to retrieve a named service account and secret
type ServiceAccountTokenGetter interface {
	GetServiceAccount(namespace, name string) (*api.ServiceAccount, error)
	GetSecret(namespace, name string) (*api.Secret, error)
}

// clientGetter implements ServiceAccountTokenGetter using a client.Interface
type clientGetter struct {
	client client.Interface
}

// NewGetterFromClient returns a ServiceAccountTokenGetter that
// uses the specified client to retrieve service accounts and secrets.
// The client should NOT authenticate using a service account token
// the returned getter will be used to retrieve, or recursion will result.
func NewGetterFromClient(c client.Interface) ServiceAccountTokenGetter {
	return clientGetter{c}
}
func (c clientGetter) GetServiceAccount(namespace, name string) (*api.ServiceAccount, error) {
	return c.client.ServiceAccounts(namespace).Get(name)
}
func (c clientGetter) GetSecret(namespace, name string) (*api.Secret, error) {
	return c.client.Secrets(namespace).Get(name)
}

// registryGetter implements ServiceAccountTokenGetter using a service account and secret registry
type registryGetter struct {
	serviceAccounts serviceaccount.Registry
	secrets         secret.Registry
}

// NewGetterFromRegistries returns a ServiceAccountTokenGetter that
// uses the specified registries to retrieve service accounts and secrets.
func NewGetterFromRegistries(serviceAccounts serviceaccount.Registry, secrets secret.Registry) ServiceAccountTokenGetter {
	return &registryGetter{serviceAccounts, secrets}
}
func (r *registryGetter) GetServiceAccount(namespace, name string) (*api.ServiceAccount, error) {
	ctx := api.WithNamespace(api.NewContext(), namespace)
	return r.serviceAccounts.GetServiceAccount(ctx, name)
}
func (r *registryGetter) GetSecret(namespace, name string) (*api.Secret, error) {
	ctx := api.WithNamespace(api.NewContext(), namespace)
	return r.secrets.GetSecret(ctx, name)
}

// NewGetterFromStorageInterface returns a ServiceAccountTokenGetter that
// uses the specified storage to retrieve service accounts and secrets.
func NewGetterFromStorageInterface(storage storage.Interface) ServiceAccountTokenGetter {
	return NewGetterFromRegistries(
		serviceaccount.NewRegistry(serviceaccountetcd.NewStorage(storage)),
		secret.NewRegistry(secretetcd.NewStorage(storage)),
	)
}
