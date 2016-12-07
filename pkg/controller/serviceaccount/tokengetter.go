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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/registry/core/secret"
	secretetcd "k8s.io/kubernetes/pkg/registry/core/secret/etcd"
	serviceaccountregistry "k8s.io/kubernetes/pkg/registry/core/serviceaccount"
	serviceaccountetcd "k8s.io/kubernetes/pkg/registry/core/serviceaccount/etcd"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/kubernetes/pkg/storage/storagebackend"
)

// clientGetter implements ServiceAccountTokenGetter using a clientset.Interface
type clientGetter struct {
	client clientset.Interface
}

// NewGetterFromClient returns a ServiceAccountTokenGetter that
// uses the specified client to retrieve service accounts and secrets.
// The client should NOT authenticate using a service account token
// the returned getter will be used to retrieve, or recursion will result.
func NewGetterFromClient(c clientset.Interface) serviceaccount.ServiceAccountTokenGetter {
	return clientGetter{c}
}
func (c clientGetter) GetServiceAccount(namespace, name string) (*v1.ServiceAccount, error) {
	return c.client.Core().ServiceAccounts(namespace).Get(name)
}
func (c clientGetter) GetSecret(namespace, name string) (*v1.Secret, error) {
	return c.client.Core().Secrets(namespace).Get(name)
}

// registryGetter implements ServiceAccountTokenGetter using a service account and secret registry
type registryGetter struct {
	serviceAccounts serviceaccountregistry.Registry
	secrets         secret.Registry
}

// NewGetterFromRegistries returns a ServiceAccountTokenGetter that
// uses the specified registries to retrieve service accounts and secrets.
func NewGetterFromRegistries(serviceAccounts serviceaccountregistry.Registry, secrets secret.Registry) serviceaccount.ServiceAccountTokenGetter {
	return &registryGetter{serviceAccounts, secrets}
}
func (r *registryGetter) GetServiceAccount(namespace, name string) (*v1.ServiceAccount, error) {
	ctx := api.WithNamespace(api.NewContext(), namespace)
	internalServiceAccount, err := r.serviceAccounts.GetServiceAccount(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	v1ServiceAccount := v1.ServiceAccount{}
	err = v1.Convert_api_ServiceAccount_To_v1_ServiceAccount(internalServiceAccount, &v1ServiceAccount, nil)
	return &v1ServiceAccount, err

}
func (r *registryGetter) GetSecret(namespace, name string) (*v1.Secret, error) {
	ctx := api.WithNamespace(api.NewContext(), namespace)
	internalSecret, err := r.secrets.GetSecret(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	v1Secret := v1.Secret{}
	err = v1.Convert_api_Secret_To_v1_Secret(internalSecret, &v1Secret, nil)
	return &v1Secret, err

}

// NewGetterFromStorageInterface returns a ServiceAccountTokenGetter that
// uses the specified storage to retrieve service accounts and secrets.
func NewGetterFromStorageInterface(config *storagebackend.Config, saPrefix, secretPrefix string) serviceaccount.ServiceAccountTokenGetter {
	return NewGetterFromRegistries(
		serviceaccountregistry.NewRegistry(serviceaccountetcd.NewREST(generic.RESTOptions{StorageConfig: config, Decorator: generic.UndecoratedStorage, ResourcePrefix: saPrefix})),
		secret.NewRegistry(secretetcd.NewREST(generic.RESTOptions{StorageConfig: config, Decorator: generic.UndecoratedStorage, ResourcePrefix: secretPrefix})),
	)
}
