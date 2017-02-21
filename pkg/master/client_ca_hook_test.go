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

package master

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
)

func TestWriteClientCAs(t *testing.T) {
	tests := []struct {
		name               string
		hook               ClientCARegistrationHook
		preexistingObjs    []runtime.Object
		expectedConfigMaps map[string]*api.ConfigMap
		expectUpdate       bool
	}{
		{
			name: "basic",
			hook: ClientCARegistrationHook{
				ClientCA:               []byte("foo"),
				FrontProxyCA:           []byte("bar"),
				FrontProxyAllowedNames: []string{"first", "second"},
			},
			expectedConfigMaps: map[string]*api.ConfigMap{
				"client-ca": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespacePublic, Name: "client-ca"},
					Data: map[string]string{
						"client-ca.crt": "foo",
					},
				},
				"front-proxy-ca": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespacePublic, Name: "front-proxy-ca"},
					Data: map[string]string{
						"front-proxy-ca.crt":        "bar",
						"front-proxy-allowed-names": `["first","second"]`,
					},
				},
			},
		},
		{
			name: "skip client-ca",
			hook: ClientCARegistrationHook{
				FrontProxyCA:           []byte("bar"),
				FrontProxyAllowedNames: []string{"first", "second"},
			},
			expectedConfigMaps: map[string]*api.ConfigMap{
				"front-proxy-ca": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespacePublic, Name: "front-proxy-ca"},
					Data: map[string]string{
						"front-proxy-ca.crt":        "bar",
						"front-proxy-allowed-names": `["first","second"]`,
					},
				},
			},
		},
		{
			name: "skip front-proxy-ca",
			hook: ClientCARegistrationHook{
				ClientCA: []byte("foo"),
			},
			expectedConfigMaps: map[string]*api.ConfigMap{
				"client-ca": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespacePublic, Name: "client-ca"},
					Data: map[string]string{
						"client-ca.crt": "foo",
					},
				},
			},
		},
		{
			name: "empty allowed names",
			hook: ClientCARegistrationHook{
				FrontProxyCA: []byte("bar"),
			},
			expectedConfigMaps: map[string]*api.ConfigMap{
				"front-proxy-ca": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespacePublic, Name: "front-proxy-ca"},
					Data: map[string]string{
						"front-proxy-ca.crt":        "bar",
						"front-proxy-allowed-names": `null`,
					},
				},
			},
		},
		{
			name: "overwrite client-ca",
			hook: ClientCARegistrationHook{
				ClientCA: []byte("foo"),
			},
			preexistingObjs: []runtime.Object{
				&api.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespacePublic, Name: "client-ca"},
					Data: map[string]string{
						"client-ca.crt": "other",
					},
				},
			},
			expectedConfigMaps: map[string]*api.ConfigMap{
				"client-ca": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespacePublic, Name: "client-ca"},
					Data: map[string]string{
						"client-ca.crt": "foo",
					},
				},
			},
			expectUpdate: true,
		},
		{
			name: "overwrite front-proxy-ca",
			hook: ClientCARegistrationHook{
				FrontProxyCA:           []byte("bar"),
				FrontProxyAllowedNames: []string{},
			},
			preexistingObjs: []runtime.Object{
				&api.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespacePublic, Name: "front-proxy-ca"},
					Data: map[string]string{
						"front-proxy-ca.crt":        "something",
						"front-proxy-allowed-names": `null`,
					},
				},
			},
			expectedConfigMaps: map[string]*api.ConfigMap{
				"front-proxy-ca": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespacePublic, Name: "front-proxy-ca"},
					Data: map[string]string{
						"front-proxy-ca.crt":        "bar",
						"front-proxy-allowed-names": `[]`,
					},
				},
			},
			expectUpdate: true,
		},
		{
			name: "namespace exists",
			hook: ClientCARegistrationHook{
				ClientCA: []byte("foo"),
			},
			preexistingObjs: []runtime.Object{
				&api.Namespace{ObjectMeta: metav1.ObjectMeta{Name: metav1.NamespacePublic}},
			},
			expectedConfigMaps: map[string]*api.ConfigMap{
				"client-ca": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespacePublic, Name: "client-ca"},
					Data: map[string]string{
						"client-ca.crt": "foo",
					},
				},
			},
		},
	}

	for _, test := range tests {
		client := fake.NewSimpleClientset(test.preexistingObjs...)
		test.hook.writeClientCAs(client.Core())

		actualConfigMaps, updated := getFinalConfiMaps(client)
		if !reflect.DeepEqual(test.expectedConfigMaps, actualConfigMaps) {
			t.Errorf("%s: %v", test.name, diff.ObjectReflectDiff(test.expectedConfigMaps, actualConfigMaps))
			continue
		}
		if test.expectUpdate != updated {
			t.Errorf("%s: expected %v, got %v", test.name, test.expectUpdate, updated)
			continue
		}
	}
}

func getFinalConfiMaps(client *fake.Clientset) (map[string]*api.ConfigMap, bool) {
	ret := map[string]*api.ConfigMap{}
	updated := false

	for _, action := range client.Actions() {
		if action.Matches("create", "configmaps") {
			obj := action.(clienttesting.CreateAction).GetObject().(*api.ConfigMap)
			ret[obj.Name] = obj
		}
		if action.Matches("update", "configmaps") {
			updated = true
			obj := action.(clienttesting.UpdateAction).GetObject().(*api.ConfigMap)
			ret[obj.Name] = obj
		}
	}
	return ret, updated
}
