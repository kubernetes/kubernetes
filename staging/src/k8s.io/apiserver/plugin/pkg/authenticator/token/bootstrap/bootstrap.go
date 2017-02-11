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

/*
Package bootstrap provides a token authenticator for TLS bootstrap secrets.
*/
package bootstrap

import (
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
)

const (
	SecretType = "bootstrap.kubernetes.io/token"

	TokenID     = "token-id"
	TokenSecret = "token-secret"

	BootstrapUserPrefix = "system:bootstrap:"
	BootstrapGroup      = "system:bootstrappers"
)

// SecretLister is a Kubernetes clients which can list secrets in a namespace.
type SecretLister interface {
	ListSecrets(namespace string) (*v1.SecretList, error)
}

// NewTokenAuthenticator initializes a bootstrap token authenticator.
func NewTokenAuthenticator(s SecretLister) *TokenAuthenticator {
	return &TokenAuthenticator{s}
}

// TokenAuthenticator authenticates bootstrap tokens from secrets in the API server.
type TokenAuthenticator struct {
	lister SecretLister

	// TODO(ericchiang): Does the SecretLister do the caching or do we do it here?
}

// AuthenticateToken tries to match the provided token to a bootstrap token secret
// in the "kube-system" namespace. If found, it authenticates the token in the
// "system:bootstrappers" group and with the "system:bootstrap:(token-id)" username.
//
// All secrets must be of type "bootstrap.kubernetes.io/token". An example secret:
//
//     apiVersion: v1
//     kind: Secret
//     metadata:
//       name: node-bootstrap-token
//       namespace: kube-system
//     data:
//       token-secret: ( bearer token )
//       token-id: ( token id )
//     type: bootstrap.kubernetes.io/token
//
func (t *TokenAuthenticator) AuthenticateToken(token string) (user.Info, bool, error) {
	secrets, err := t.lister.ListSecrets(api.NamespaceSystem)
	if err != nil {
		return nil, false, err
	}

	for _, secret := range secrets.Items {
		if secret.Type != SecretType || secret.Data == nil {
			continue
		}

		if ts, ok := secret.Data[TokenSecret]; !ok || token != string(ts) {
			continue
		}

		id, ok := secret.Data[TokenID]
		if !ok || len(id) == 0 {
			continue
		}

		return &user.DefaultInfo{
			Name:   BootstrapUserPrefix + string(id),
			Groups: []string{BootstrapGroup},
		}, true, nil
	}
	return nil, false, nil
}
