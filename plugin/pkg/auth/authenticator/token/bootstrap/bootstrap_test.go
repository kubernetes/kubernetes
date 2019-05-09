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
	"context"
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/user"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
)

type lister struct {
	secrets []*corev1.Secret
}

func (l *lister) List(selector labels.Selector) (ret []*corev1.Secret, err error) {
	return l.secrets, nil
}

func (l *lister) Get(name string) (*corev1.Secret, error) {
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

		secrets []*corev1.Secret
		token   string

		wantNotFound bool
		wantUser     *user.DefaultInfo
	}{
		{
			name: "valid token",
			secrets: []*corev1.Secret{
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
			secrets: []*corev1.Secret{
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
			secrets: []*corev1.Secret{
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
			secrets: []*corev1.Secret{
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
			secrets: []*corev1.Secret{
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
			secrets: []*corev1.Secret{
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
			secrets: []*corev1.Secret{
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
			secrets: []*corev1.Secret{
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
			secrets: []*corev1.Secret{
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
			secrets: []*corev1.Secret{
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
			resp, found, err := a.AuthenticateToken(context.Background(), test.token)
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

			gotUser := resp.User.(*user.DefaultInfo)
			if !reflect.DeepEqual(gotUser, test.wantUser) {
				t.Errorf("test %q want user=%#v, got=%#v", test.name, test.wantUser, gotUser)
			}
		}()
	}
}

func TestGetGroups(t *testing.T) {
	tests := []struct {
		name         string
		secret       *corev1.Secret
		expectResult []string
		expectError  bool
	}{
		{
			name: "not set",
			secret: &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
				Data:       map[string][]byte{},
			},
			expectResult: []string{"system:bootstrappers"},
		},
		{
			name: "set to empty value",
			secret: &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
				Data: map[string][]byte{
					bootstrapapi.BootstrapTokenExtraGroupsKey: []byte(""),
				},
			},
			expectResult: []string{"system:bootstrappers"},
		},
		{
			name: "invalid prefix",
			secret: &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
				Data: map[string][]byte{
					bootstrapapi.BootstrapTokenExtraGroupsKey: []byte("foo"),
				},
			},
			expectError: true,
		},
		{
			name: "valid",
			secret: &corev1.Secret{
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
