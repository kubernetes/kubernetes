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
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/kubernetes/pkg/api"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
)

type lister struct {
	secrets []*api.Secret
}

func (l *lister) List(selector labels.Selector) (ret []*api.Secret, err error) {
	return l.secrets, nil
}

func (l *lister) Get(name string) (*api.Secret, error) {
	for _, s := range l.secrets {
		if s.Name == name {
			return s, nil
		}
	}
	return nil, errors.NewNotFound(schema.GroupResource{}, name)
}

const (
	tokenID     = "foobar"           // 6 letters
	tokenSecret = "circumnavigation" // 16 letters
)

func TestTokenAuthenticator(t *testing.T) {
	now := metav1.Now()

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
					ObjectMeta: metav1.ObjectMeta{
						Name: bootstrapapi.BootstrapTokenSecretPrefix + tokenID,
					},
					Data: map[string][]byte{
						bootstrapapi.BootstrapTokenIDKey:               []byte(tokenID),
						bootstrapapi.BootstrapTokenSecretKey:           []byte(tokenSecret),
						bootstrapapi.BootstrapTokenUsageAuthentication: []byte("true"),
					},
					Type: "bootstrap.kubernetes.io/token",
				},
			},
			token: tokenID + "." + tokenSecret,
			wantUser: &user.DefaultInfo{
				Name:   "system:bootstrap:" + tokenID,
				Groups: []string{"system:bootstrappers"},
			},
		},
		{
			name: "valid token with extra group",
			secrets: []*api.Secret{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: bootstrapapi.BootstrapTokenSecretPrefix + tokenID,
					},
					Data: map[string][]byte{
						bootstrapapi.BootstrapTokenIDKey:               []byte(tokenID),
						bootstrapapi.BootstrapTokenSecretKey:           []byte(tokenSecret),
						bootstrapapi.BootstrapTokenUsageAuthentication: []byte("true"),
						bootstrapapi.BootstrapTokenExtraGroupsKey:      []byte("system:bootstrappers:foo"),
					},
					Type: "bootstrap.kubernetes.io/token",
				},
			},
			token: tokenID + "." + tokenSecret,
			wantUser: &user.DefaultInfo{
				Name:   "system:bootstrap:" + tokenID,
				Groups: []string{"system:bootstrappers", "system:bootstrappers:foo"},
			},
		},
		{
			name: "invalid group",
			secrets: []*api.Secret{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: bootstrapapi.BootstrapTokenSecretPrefix + tokenID,
					},
					Data: map[string][]byte{
						bootstrapapi.BootstrapTokenIDKey:               []byte(tokenID),
						bootstrapapi.BootstrapTokenSecretKey:           []byte(tokenSecret),
						bootstrapapi.BootstrapTokenUsageAuthentication: []byte("true"),
						bootstrapapi.BootstrapTokenExtraGroupsKey:      []byte("foo"),
					},
					Type: "bootstrap.kubernetes.io/token",
				},
			},
			token:        tokenID + "." + tokenSecret,
			wantNotFound: true,
		},
		{
			name: "invalid secret name",
			secrets: []*api.Secret{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "bad-name",
					},
					Data: map[string][]byte{
						bootstrapapi.BootstrapTokenIDKey:               []byte(tokenID),
						bootstrapapi.BootstrapTokenSecretKey:           []byte(tokenSecret),
						bootstrapapi.BootstrapTokenUsageAuthentication: []byte("true"),
					},
					Type: "bootstrap.kubernetes.io/token",
				},
			},
			token:        tokenID + "." + tokenSecret,
			wantNotFound: true,
		},
		{
			name: "no usage",
			secrets: []*api.Secret{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: bootstrapapi.BootstrapTokenSecretPrefix + tokenID,
					},
					Data: map[string][]byte{
						bootstrapapi.BootstrapTokenIDKey:     []byte(tokenID),
						bootstrapapi.BootstrapTokenSecretKey: []byte(tokenSecret),
					},
					Type: "bootstrap.kubernetes.io/token",
				},
			},
			token:        tokenID + "." + tokenSecret,
			wantNotFound: true,
		},
		{
			name: "wrong token",
			secrets: []*api.Secret{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: bootstrapapi.BootstrapTokenSecretPrefix + tokenID,
					},
					Data: map[string][]byte{
						bootstrapapi.BootstrapTokenIDKey:               []byte(tokenID),
						bootstrapapi.BootstrapTokenSecretKey:           []byte(tokenSecret),
						bootstrapapi.BootstrapTokenUsageAuthentication: []byte("true"),
					},
					Type: "bootstrap.kubernetes.io/token",
				},
			},
			token:        "barfoo" + "." + tokenSecret,
			wantNotFound: true,
		},
		{
			name: "deleted token",
			secrets: []*api.Secret{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              bootstrapapi.BootstrapTokenSecretPrefix + tokenID,
						DeletionTimestamp: &now,
					},
					Data: map[string][]byte{
						bootstrapapi.BootstrapTokenIDKey:               []byte(tokenID),
						bootstrapapi.BootstrapTokenSecretKey:           []byte(tokenSecret),
						bootstrapapi.BootstrapTokenUsageAuthentication: []byte("true"),
					},
					Type: "bootstrap.kubernetes.io/token",
				},
			},
			token:        tokenID + "." + tokenSecret,
			wantNotFound: true,
		},
		{
			name: "expired token",
			secrets: []*api.Secret{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: bootstrapapi.BootstrapTokenSecretPrefix + tokenID,
					},
					Data: map[string][]byte{
						bootstrapapi.BootstrapTokenIDKey:               []byte(tokenID),
						bootstrapapi.BootstrapTokenSecretKey:           []byte(tokenSecret),
						bootstrapapi.BootstrapTokenUsageAuthentication: []byte("true"),
						bootstrapapi.BootstrapTokenExpirationKey:       []byte("2009-11-10T23:00:00Z"),
					},
					Type: "bootstrap.kubernetes.io/token",
				},
			},
			token:        tokenID + "." + tokenSecret,
			wantNotFound: true,
		},
		{
			name: "not expired token",
			secrets: []*api.Secret{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: bootstrapapi.BootstrapTokenSecretPrefix + tokenID,
					},
					Data: map[string][]byte{
						bootstrapapi.BootstrapTokenIDKey:               []byte(tokenID),
						bootstrapapi.BootstrapTokenSecretKey:           []byte(tokenSecret),
						bootstrapapi.BootstrapTokenUsageAuthentication: []byte("true"),
						bootstrapapi.BootstrapTokenExpirationKey:       []byte("2109-11-10T23:00:00Z"),
					},
					Type: "bootstrap.kubernetes.io/token",
				},
			},
			token: tokenID + "." + tokenSecret,
			wantUser: &user.DefaultInfo{
				Name:   "system:bootstrap:" + tokenID,
				Groups: []string{"system:bootstrappers"},
			},
		},
		{
			name: "token id wrong length",
			secrets: []*api.Secret{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: bootstrapapi.BootstrapTokenSecretPrefix + "foo",
					},
					Data: map[string][]byte{
						bootstrapapi.BootstrapTokenIDKey:               []byte("foo"),
						bootstrapapi.BootstrapTokenSecretKey:           []byte(tokenSecret),
						bootstrapapi.BootstrapTokenUsageAuthentication: []byte("true"),
					},
					Type: "bootstrap.kubernetes.io/token",
				},
			},
			// Token ID must be 6 characters.
			token:        "foo" + "." + tokenSecret,
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

func TestGetGroups(t *testing.T) {
	tests := []struct {
		name         string
		secret       *api.Secret
		expectResult []string
		expectError  bool
	}{
		{
			name: "not set",
			secret: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
				Data:       map[string][]byte{},
			},
			expectResult: []string{"system:bootstrappers"},
		},
		{
			name: "set to empty value",
			secret: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
				Data: map[string][]byte{
					bootstrapapi.BootstrapTokenExtraGroupsKey: []byte(""),
				},
			},
			expectResult: []string{"system:bootstrappers"},
		},
		{
			name: "invalid prefix",
			secret: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
				Data: map[string][]byte{
					bootstrapapi.BootstrapTokenExtraGroupsKey: []byte("foo"),
				},
			},
			expectError: true,
		},
		{
			name: "valid",
			secret: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
				Data: map[string][]byte{
					bootstrapapi.BootstrapTokenExtraGroupsKey: []byte("system:bootstrappers:foo,system:bootstrappers:bar,system:bootstrappers:bar"),
				},
			},
			// expect the results in deduplicated, sorted order
			expectResult: []string{
				"system:bootstrappers",
				"system:bootstrappers:bar",
				"system:bootstrappers:foo",
			},
		},
	}
	for _, test := range tests {
		result, err := getGroups(test.secret)
		if test.expectError {
			if err == nil {
				t.Errorf("test %q expected an error, but didn't get one (result: %#v)", test.name, result)
			}
			continue
		}
		if err != nil {
			t.Errorf("test %q return an unexpected error: %v", test.name, err)
			continue
		}
		if !reflect.DeepEqual(result, test.expectResult) {
			t.Errorf("test %q expected %#v, got %#v", test.name, test.expectResult, result)
		}
	}
}
