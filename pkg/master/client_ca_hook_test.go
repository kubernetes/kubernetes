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

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
)

func TestWriteClientCAs(t *testing.T) {
	tests := []struct {
		name               string
		hook               ClientCARegistrationHook
		preexistingObjs    []runtime.Object
		expectedConfigMaps map[string]*corev1.ConfigMap
		expectUpdate       bool
	}{
		{
			name: "basic",
			hook: ClientCARegistrationHook{
				ClientCA:                         []byte("foo"),
				RequestHeaderUsernameHeaders:     []string{"alfa", "bravo", "charlie"},
				RequestHeaderGroupHeaders:        []string{"delta"},
				RequestHeaderExtraHeaderPrefixes: []string{"echo", "foxtrot"},
				RequestHeaderCA:                  []byte("bar"),
				RequestHeaderAllowedNames:        []string{"first", "second"},
			},
			expectedConfigMaps: map[string]*corev1.ConfigMap{
				"extension-apiserver-authentication": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"client-ca-file":                     "foo",
						"requestheader-username-headers":     `["alfa","bravo","charlie"]`,
						"requestheader-group-headers":        `["delta"]`,
						"requestheader-extra-headers-prefix": `["echo","foxtrot"]`,
						"requestheader-client-ca-file":       "bar",
						"requestheader-allowed-names":        `["first","second"]`,
					},
				},
			},
		},
		{
			name: "skip extension-apiserver-authentication",
			hook: ClientCARegistrationHook{
				RequestHeaderCA:           []byte("bar"),
				RequestHeaderAllowedNames: []string{"first", "second"},
			},
			expectedConfigMaps: map[string]*corev1.ConfigMap{
				"extension-apiserver-authentication": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"requestheader-username-headers":     `null`,
						"requestheader-group-headers":        `null`,
						"requestheader-extra-headers-prefix": `null`,
						"requestheader-client-ca-file":       "bar",
						"requestheader-allowed-names":        `["first","second"]`,
					},
				},
			},
		},
		{
			name: "skip extension-apiserver-authentication",
			hook: ClientCARegistrationHook{
				ClientCA: []byte("foo"),
			},
			expectedConfigMaps: map[string]*corev1.ConfigMap{
				"extension-apiserver-authentication": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"client-ca-file": "foo",
					},
				},
			},
		},
		{
			name: "empty allowed names",
			hook: ClientCARegistrationHook{
				RequestHeaderCA: []byte("bar"),
			},
			expectedConfigMaps: map[string]*corev1.ConfigMap{
				"extension-apiserver-authentication": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"requestheader-username-headers":     `null`,
						"requestheader-group-headers":        `null`,
						"requestheader-extra-headers-prefix": `null`,
						"requestheader-client-ca-file":       "bar",
						"requestheader-allowed-names":        `null`,
					},
				},
			},
		},
		{
			name: "overwrite extension-apiserver-authentication",
			hook: ClientCARegistrationHook{
				ClientCA: []byte("foo"),
			},
			preexistingObjs: []runtime.Object{
				&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"client-ca-file": "other",
					},
				},
			},
			expectedConfigMaps: map[string]*corev1.ConfigMap{
				"extension-apiserver-authentication": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"client-ca-file": "foo",
					},
				},
			},
			expectUpdate: true,
		},
		{
			name: "overwrite extension-apiserver-authentication requestheader",
			hook: ClientCARegistrationHook{
				RequestHeaderUsernameHeaders:     []string{},
				RequestHeaderGroupHeaders:        []string{},
				RequestHeaderExtraHeaderPrefixes: []string{},
				RequestHeaderCA:                  []byte("bar"),
				RequestHeaderAllowedNames:        []string{},
			},
			preexistingObjs: []runtime.Object{
				&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"requestheader-username-headers":     `null`,
						"requestheader-group-headers":        `null`,
						"requestheader-extra-headers-prefix": `null`,
						"requestheader-client-ca-file":       "something",
						"requestheader-allowed-names":        `null`,
					},
				},
			},
			expectedConfigMaps: map[string]*corev1.ConfigMap{
				"extension-apiserver-authentication": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"requestheader-username-headers":     `[]`,
						"requestheader-group-headers":        `[]`,
						"requestheader-extra-headers-prefix": `[]`,
						"requestheader-client-ca-file":       "bar",
						"requestheader-allowed-names":        `[]`,
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
				&corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: metav1.NamespaceSystem}},
			},
			expectedConfigMaps: map[string]*corev1.ConfigMap{
				"extension-apiserver-authentication": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"client-ca-file": "foo",
					},
				},
			},
		},
		{
			name: "skip on no change",
			hook: ClientCARegistrationHook{
				RequestHeaderUsernameHeaders:     []string{},
				RequestHeaderGroupHeaders:        []string{},
				RequestHeaderExtraHeaderPrefixes: []string{},
				RequestHeaderCA:                  []byte("bar"),
				RequestHeaderAllowedNames:        []string{},
			},
			preexistingObjs: []runtime.Object{
				&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"requestheader-username-headers":     `[]`,
						"requestheader-group-headers":        `[]`,
						"requestheader-extra-headers-prefix": `[]`,
						"requestheader-client-ca-file":       "bar",
						"requestheader-allowed-names":        `[]`,
					},
				},
			},
			expectedConfigMaps: map[string]*corev1.ConfigMap{},
			expectUpdate:       false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := fake.NewSimpleClientset(test.preexistingObjs...)
			test.hook.tryToWriteClientCAs(client.CoreV1())

			actualConfigMaps, updated := getFinalConfigMaps(client)
			if !reflect.DeepEqual(test.expectedConfigMaps, actualConfigMaps) {
				t.Fatalf("%s: %v", test.name, diff.ObjectReflectDiff(test.expectedConfigMaps, actualConfigMaps))
			}
			if test.expectUpdate != updated {
				t.Fatalf("%s: expected %v, got %v", test.name, test.expectUpdate, updated)
			}
		})
	}
}

func getFinalConfigMaps(client *fake.Clientset) (map[string]*corev1.ConfigMap, bool) {
	ret := map[string]*corev1.ConfigMap{}
	updated := false

	for _, action := range client.Actions() {
		if action.Matches("create", "configmaps") {
			obj := action.(clienttesting.CreateAction).GetObject().(*corev1.ConfigMap)
			ret[obj.Name] = obj
		}
		if action.Matches("update", "configmaps") {
			updated = true
			obj := action.(clienttesting.UpdateAction).GetObject().(*corev1.ConfigMap)
			ret[obj.Name] = obj
		}
	}
	return ret, updated
}
