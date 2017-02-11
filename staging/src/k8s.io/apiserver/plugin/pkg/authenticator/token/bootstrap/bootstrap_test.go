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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	"k8s.io/kubernetes/pkg/controller"
)

func TestTokenAuthenticator(t *testing.T) {
	tests := []struct {
		name string

		secrets []runtime.Object
		token   string

		wantNotFound bool
		wantUser     *user.DefaultInfo
	}{
		{
			name: "valid token",
			secrets: []runtime.Object{
				&api.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "foo",
						Namespace: "kube-system",
					},
					Data: map[string][]byte{
						TokenID:     []byte("node1"),
						TokenSecret: []byte("foobar"),
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
			name: "wrong namespace",
			secrets: []runtime.Object{
				&api.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "foo",
						Namespace: "wrong-namespace",
					},
					Data: map[string][]byte{
						TokenID:     []byte("node1"),
						TokenSecret: []byte("foobar"),
					},
					Type: SecretType,
				},
			},
			token:        "node1:foobar",
			wantNotFound: true,
		},
		{
			name: "wrong token",
			secrets: []runtime.Object{
				&api.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "foo",
						Namespace: "kube-system",
					},
					Data: map[string][]byte{
						TokenID:     []byte("node1"),
						TokenSecret: []byte("foobar"),
					},
					Type: SecretType,
				},
			},
			token:        "node1:barfoo",
			wantNotFound: true,
		},
	}

	for _, test := range tests {
		func() {
			f := internalversion.NewSharedInformerFactory(
				fake.NewSimpleClientset(test.secrets...),
				controller.NoResyncPeriodFunc(),
			)
			informer := f.Core().InternalVersion().Secrets()
			a := NewTokenAuthenticator(informer)

			c := make(chan struct{})
			f.Start(c)
			defer close(c)

			// Without this sleep, the informer doesn't do its initial sync.
			time.Sleep(5 * time.Millisecond)

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
