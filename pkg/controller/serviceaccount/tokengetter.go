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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/registry/core/secret"
	secretstore "k8s.io/kubernetes/pkg/registry/core/secret/storage"
	serviceaccountregistry "k8s.io/kubernetes/pkg/registry/core/serviceaccount"
	serviceaccountstore "k8s.io/kubernetes/pkg/registry/core/serviceaccount/storage"
	"k8s.io/kubernetes/pkg/serviceaccount"
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
	return c.client.Core().ServiceAccounts(namespace).Get(name, metav1.GetOptions{})
}
func (c clientGetter) GetSecret(namespace, name string) (*v1.Secret, error) {
	return c.client.Core().Secrets(namespace).Get(name, metav1.GetOptions{})
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
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)
	internalServiceAccount, err := r.serviceAccounts.GetServiceAccount(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	v1ServiceAccount := v1.ServiceAccount{}
	err = v1.Convert_api_ServiceAccount_To_v1_ServiceAccount(internalServiceAccount, &v1ServiceAccount, nil)
	return &v1ServiceAccount, err

}
func (r *registryGetter) GetSecret(namespace, name string) (*v1.Secret, error) {
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)
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
func NewGetterFromStorageInterface(
	saConfig *storagebackend.Config,
	saPrefix string,
	secretConfig *storagebackend.Config,
	secretPrefix string) serviceaccount.ServiceAccountTokenGetter {

	saOpts := generic.RESTOptions{StorageConfig: saConfig, Decorator: generic.UndecoratedStorage, ResourcePrefix: saPrefix}
	secretOpts := generic.RESTOptions{StorageConfig: secretConfig, Decorator: generic.UndecoratedStorage, ResourcePrefix: secretPrefix}
	return NewGetterFromRegistries(
		serviceaccountregistry.NewRegistry(serviceaccountstore.NewREST(saOpts)),
		secret.NewRegistry(secretstore.NewREST(secretOpts)),
	)
}
