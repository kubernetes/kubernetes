/*
Copyright 2017 The Kubernetes Authors.

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

package bootstrap

import (
	"errors"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/kubernetes/pkg/api"
)

type lister struct {
	secrets []*api.Secret
}

func (l *lister) List(selector labels.Selector) (ret []*api.Secret, err error) {
	// Authenticator doesn't use label selectors.
	return l.secrets, nil
}

func (l *lister) Get(name string) (*api.Secret, error) {
	return nil, errors.New("not implemented")
}

func TestTokenAuthenticator(t *testing.T) {
	tests := []struct {
		name string

		secrets []*api.Secret
		token   string

		wantNotFound bool
		wantUser     *user.DefaultInfo
	}{
		{
			name: "valid token",
			secrets: []*api.Secret{
				{
					Data: map[string][]byte{
						BootstrapTokenID:                  []byte("node1"),
						BootstrapTokenSecret:              []byte("foobar"),
						BootstrapTokenUsageAuthentication: []byte("true"),
					},
					Type: SecretType,
				},
			},
			token: "node1:foobar",
			wantUser: &user.DefaultInfo{
				Name:   "system:bootstrap:node1",
				Groups: []string{"system:bootstrappers"},
			},
		},
		{
			name: "no usage",
			secrets: []*api.Secret{
				&api.Secret{
					Data: map[string][]byte{
						BootstrapTokenID:     []byte("node1"),
						BootstrapTokenSecret: []byte("foobar"),
					},
					Type: SecretType,
				},
			},
			token:        "node1:foobar",
			wantNotFound: true,
		},
		{
			name: "wrong token",
			secrets: []*api.Secret{
				&api.Secret{
					Data: map[string][]byte{
						BootstrapTokenID:                  []byte("node1"),
						BootstrapTokenSecret:              []byte("foobar"),
						BootstrapTokenUsageAuthentication: []byte("true"),
					},
					Type: SecretType,
				},
			},
			token:        "node1:barfoo",
			wantNotFound: true,
		},
		{
			name: "expired token",
			secrets: []*api.Secret{
				{
					Data: map[string][]byte{
						BootstrapTokenID:                  []byte("node1"),
						BootstrapTokenSecret:              []byte("foobar"),
						BootstrapTokenUsageAuthentication: []byte("true"),
						BootstrapTokenExpirationKey:       []byte("2009-11-10T23:00:00Z"),
					},
					Type: SecretType,
				},
			},
			token:        "node1:foobar",
			wantNotFound: true,
		},
	}

	for _, test := range tests {
		func() {
			a := NewTokenAuthenticator(&lister{test.secrets})
			u, found, err := a.AuthenticateToken(test.token)
			if err != nil {
				t.Errorf("test %q returned an error: %v", test.name, err)
				return
			}

			if !found {
				if !test.wantNotFound {
					t.Errorf("test %q expected to get user", test.name)
				}
				return
			}

			if test.wantNotFound {
				t.Errorf("test %q expected to not get a user", test.name)
				return
			}

			gotUser := u.(*user.DefaultInfo)

			if !reflect.DeepEqual(gotUser, test.wantUser) {
				t.Errorf("test %q want user=%#v, got=%#v", test.name, test.wantUser, gotUser)
			}
		}()
	}
}
