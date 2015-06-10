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

package etcd

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/secret"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/serviceaccount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
)

// REST implements a RESTStorage for service accounts against etcd
type REST struct {
	*etcdgeneric.Etcd
}

const Prefix = "/serviceaccounts"

// NewStorage returns a RESTStorage object that will work against service accounts objects.
func NewStorage(h tools.EtcdHelper) *REST {
	store := &etcdgeneric.Etcd{
		NewFunc:     func() runtime.Object { return &api.ServiceAccount{} },
		NewListFunc: func() runtime.Object { return &api.ServiceAccountList{} },
		KeyRootFunc: func(ctx api.Context) string {
			return etcdgeneric.NamespaceKeyRootFunc(ctx, Prefix)
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return etcdgeneric.NamespaceKeyFunc(ctx, Prefix, name)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.ServiceAccount).Name, nil
		},
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return serviceaccount.Matcher(label, field)
		},
		EndpointName: "serviceaccounts",

		Helper: h,
	}

	store.CreateStrategy = serviceaccount.Strategy
	store.UpdateStrategy = serviceaccount.Strategy
	store.ReturnDeletedObject = true

	return &REST{store}
}

func NewTokenStorage(serviceAccounts serviceaccount.Registry, secrets secret.Registry, generator serviceaccount.TokenGenerator) *TokenREST {
	return &TokenREST{
		serviceAccounts: serviceAccounts,
		secrets:         secrets,
		generator:       generator,
	}
}

var _ = rest.NamedCreater(&TokenREST{})

// TokenREST implements rest.NamedCreater for secrets of type ServiceAccountToken
type TokenREST struct {
	serviceAccounts serviceaccount.Registry
	secrets         secret.Registry
	generator       serviceaccount.TokenGenerator
}

// New returns an empty ServiceAccountTokenRequest object
func (t *TokenREST) New() runtime.Object {
	return &api.ServiceAccountTokenRequest{}
}

// Create
func (t *TokenREST) Create(ctx api.Context, serviceAccountName string, obj runtime.Object) (runtime.Object, error) {
	// Ensure we are enabled
	if t.generator == nil {
		return nil, errors.NewInternalError(fmt.Errorf("Token generation is not enabled"))
	}

	// Ensure correct type
	tokenRequest, ok := obj.(*api.ServiceAccountTokenRequest)
	if !ok {
		return nil, fmt.Errorf("Invalid type: %#v", obj)
	}

	// Look up the service account
	serviceAccount, err := t.serviceAccounts.GetServiceAccount(ctx, serviceAccountName)
	if err != nil {
		return nil, err
	}

	// Build the secret
	secret := &api.Secret{
		ObjectMeta: api.ObjectMeta{
			// Pre-generate the name so it is available to the token generator
			Name:      secret.Strategy.GenerateName(fmt.Sprintf("%s-token-", serviceAccount.Name)),
			Namespace: serviceAccount.Namespace,
			Annotations: map[string]string{
				api.ServiceAccountNameKey: serviceAccount.Name,
				api.ServiceAccountUIDKey:  string(serviceAccount.UID),
			},
		},
		Type: api.SecretTypeServiceAccountToken,
		Data: map[string][]byte{},
	}

	// Generate the token
	token, err := t.generator.GenerateToken(*tokenRequest, *serviceAccount, *secret)
	if err != nil {
		return nil, err
	}
	secret.Data[api.ServiceAccountTokenKey] = []byte(token)

	// Save the secret
	createdSecret, err := t.secrets.CreateSecret(ctx, secret)
	if err != nil {
		return nil, err
	}

	// Build the response
	tokenResponse := &api.ServiceAccountTokenResponse{
		Secret: api.LocalObjectReference{
			Name: createdSecret.Name,
		},
	}

	return tokenResponse, nil
}
