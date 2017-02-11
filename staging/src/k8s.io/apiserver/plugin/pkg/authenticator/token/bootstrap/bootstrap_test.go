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

package bootstrap

import (
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/kubernetes/pkg/api/v1"
)

type secretLister struct {
	secrets []v1.Secret
}

func (s secretLister) ListSecrets(namespace string) (*v1.SecretList, error) {
	var list v1.SecretList
	if namespace == "kube-system" {
		list.Items = s.secrets
	}
	return &list, nil
}

func TestTokenAuthenticator(t *testing.T) {
	tests := []struct {
		name string

		secrets []v1.Secret
		token   string

		wantNotFound bool
		wantUser     *user.DefaultInfo
	}{
		{
			name: "valid token",
			secrets: []v1.Secret{
				{
					Data: map[string][]byte{
						TokenID:     []byte("node1"),
						TokenSecret: []byte("foobar"),
					},
					Type: SecretType,
				},
			},
			token: "foobar",
			wantUser: &user.DefaultInfo{
				Name:   "system:bootstrap:node1",
				Groups: []string{"system:bootstrappers"},
			},
		},
	}

	for _, test := range tests {
		a := NewTokenAuthenticator(&secretLister{test.secrets})
		u, found, err := a.AuthenticateToken(test.token)
		if err != nil {
			t.Errorf("test %q returned an error: %v", test.name, err)
			continue
		}

		if !found {
			if !test.wantNotFound {
				t.Errorf("test %q expected to get user", test.name)
			}
			continue
		}

		if test.wantNotFound {
			t.Errorf("test %q expected to not get a user", test.name)
			continue
		}

		gotUser := u.(*user.DefaultInfo)

		if !reflect.DeepEqual(gotUser, test.wantUser) {
			t.Errorf("test %q want user=%#v, got=%#v", test.name, test.wantUser, gotUser)
		}
	}
}
