/*
Copyright 2019 The Kubernetes Authors.

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

package clusterauthenticationtrust

import (
	"reflect"
	"testing"

	"github.com/davecgh/go-spew/spew"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authentication/request/headerrequest"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	"k8s.io/client-go/kubernetes/fake"
	corev1listers "k8s.io/client-go/listers/core/v1"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
)

var (
	someRandomCA = []byte(`-----BEGIN CERTIFICATE-----
MIIBqDCCAU2gAwIBAgIUfbqeieihh/oERbfvRm38XvS/xHAwCgYIKoZIzj0EAwIw
GjEYMBYGA1UEAxMPSW50ZXJtZWRpYXRlLUNBMCAXDTE2MTAxMTA1MDYwMFoYDzIx
MTYwOTE3MDUwNjAwWjAUMRIwEAYDVQQDEwlNeSBDbGllbnQwWTATBgcqhkjOPQIB
BggqhkjOPQMBBwNCAARv6N4R/sjMR65iMFGNLN1GC/vd7WhDW6J4X/iAjkRLLnNb
KbRG/AtOUZ+7upJ3BWIRKYbOabbQGQe2BbKFiap4o3UwczAOBgNVHQ8BAf8EBAMC
BaAwEwYDVR0lBAwwCgYIKwYBBQUHAwIwDAYDVR0TAQH/BAIwADAdBgNVHQ4EFgQU
K/pZOWpNcYai6eHFpmJEeFpeQlEwHwYDVR0jBBgwFoAUX6nQlxjfWnP6aM1meO/Q
a6b3a9kwCgYIKoZIzj0EAwIDSQAwRgIhAIWTKw/sjJITqeuNzJDAKU4xo1zL+xJ5
MnVCuBwfwDXCAiEAw/1TA+CjPq9JC5ek1ifR0FybTURjeQqYkKpve1dveps=
-----END CERTIFICATE-----
`)
	anotherRandomCA = []byte(`-----BEGIN CERTIFICATE-----
MIIDQDCCAiigAwIBAgIJANWw74P5KJk2MA0GCSqGSIb3DQEBCwUAMDQxMjAwBgNV
BAMMKWdlbmVyaWNfd2ViaG9va19hZG1pc3Npb25fcGx1Z2luX3Rlc3RzX2NhMCAX
DTE3MTExNjAwMDUzOVoYDzIyOTEwOTAxMDAwNTM5WjAjMSEwHwYDVQQDExh3ZWJo
b29rLXRlc3QuZGVmYXVsdC5zdmMwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEK
AoIBAQDXd/nQ89a5H8ifEsigmMd01Ib6NVR3bkJjtkvYnTbdfYEBj7UzqOQtHoLa
dIVmefny5uIHvj93WD8WDVPB3jX2JHrXkDTXd/6o6jIXHcsUfFTVLp6/bZ+Anqe0
r/7hAPkzA2A7APyTWM3ZbEeo1afXogXhOJ1u/wz0DflgcB21gNho4kKTONXO3NHD
XLpspFqSkxfEfKVDJaYAoMnYZJtFNsa2OvsmLnhYF8bjeT3i07lfwrhUZvP+7Gsp
7UgUwc06WuNHjfx1s5e6ySzH0QioMD1rjYneqOvk0pKrMIhuAEWXqq7jlXcDtx1E
j+wnYbVqqVYheHZ8BCJoVAAQGs9/AgMBAAGjZDBiMAkGA1UdEwQCMAAwCwYDVR0P
BAQDAgXgMB0GA1UdJQQWMBQGCCsGAQUFBwMCBggrBgEFBQcDATApBgNVHREEIjAg
hwR/AAABghh3ZWJob29rLXRlc3QuZGVmYXVsdC5zdmMwDQYJKoZIhvcNAQELBQAD
ggEBAD/GKSPNyQuAOw/jsYZesb+RMedbkzs18sSwlxAJQMUrrXwlVdHrA8q5WhE6
ABLqU1b8lQ8AWun07R8k5tqTmNvCARrAPRUqls/ryER+3Y9YEcxEaTc3jKNZFLbc
T6YtcnkdhxsiO136wtiuatpYL91RgCmuSpR8+7jEHhuFU01iaASu7ypFrUzrKHTF
bKwiLRQi1cMzVcLErq5CDEKiKhUkoDucyARFszrGt9vNIl/YCcBOkcNvM3c05Hn3
M++C29JwS3Hwbubg6WO3wjFjoEhpCwU6qRYUz3MRp4tHO4kxKXx+oQnUiFnR7vW0
YkNtGc1RUDHwecCTFpJtPb7Yu/E=
-----END CERTIFICATE-----
`)

	someRandomCAProvider    dynamiccertificates.CAContentProvider
	anotherRandomCAProvider dynamiccertificates.CAContentProvider
)

func init() {
	var err error
	someRandomCAProvider, err = dynamiccertificates.NewStaticCAContent("foo", someRandomCA)
	if err != nil {
		panic(err)
	}
	anotherRandomCAProvider, err = dynamiccertificates.NewStaticCAContent("bar", anotherRandomCA)
	if err != nil {
		panic(err)
	}
}

func TestWriteClientCAs(t *testing.T) {
	tests := []struct {
		name               string
		clusterAuthInfo    ClusterAuthenticationInfo
		preexistingObjs    []runtime.Object
		expectedConfigMaps map[string]*corev1.ConfigMap
		expectCreate       bool
	}{
		{
			name: "basic",
			clusterAuthInfo: ClusterAuthenticationInfo{
				ClientCA:                         someRandomCAProvider,
				RequestHeaderUsernameHeaders:     headerrequest.StaticStringSlice{"alfa", "bravo", "charlie"},
				RequestHeaderGroupHeaders:        headerrequest.StaticStringSlice{"delta"},
				RequestHeaderExtraHeaderPrefixes: headerrequest.StaticStringSlice{"echo", "foxtrot"},
				RequestHeaderCA:                  anotherRandomCAProvider,
				RequestHeaderAllowedNames:        headerrequest.StaticStringSlice{"first", "second"},
			},
			expectedConfigMaps: map[string]*corev1.ConfigMap{
				"extension-apiserver-authentication": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"client-ca-file":                     string(someRandomCA),
						"requestheader-username-headers":     `["alfa","bravo","charlie"]`,
						"requestheader-group-headers":        `["delta"]`,
						"requestheader-extra-headers-prefix": `["echo","foxtrot"]`,
						"requestheader-client-ca-file":       string(anotherRandomCA),
						"requestheader-allowed-names":        `["first","second"]`,
					},
				},
			},
			expectCreate: true,
		},
		{
			name: "skip extension-apiserver-authentication",
			clusterAuthInfo: ClusterAuthenticationInfo{
				RequestHeaderCA:           anotherRandomCAProvider,
				RequestHeaderAllowedNames: headerrequest.StaticStringSlice{"first", "second"},
			},
			expectedConfigMaps: map[string]*corev1.ConfigMap{
				"extension-apiserver-authentication": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"requestheader-username-headers":     `[]`,
						"requestheader-group-headers":        `[]`,
						"requestheader-extra-headers-prefix": `[]`,
						"requestheader-client-ca-file":       string(anotherRandomCA),
						"requestheader-allowed-names":        `["first","second"]`,
					},
				},
			},
			expectCreate: true,
		},
		{
			name: "skip extension-apiserver-authentication",
			clusterAuthInfo: ClusterAuthenticationInfo{
				ClientCA: someRandomCAProvider,
			},
			expectedConfigMaps: map[string]*corev1.ConfigMap{
				"extension-apiserver-authentication": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"client-ca-file": string(someRandomCA),
					},
				},
			},
			expectCreate: true,
		},
		{
			name: "empty allowed names",
			clusterAuthInfo: ClusterAuthenticationInfo{
				RequestHeaderCA: anotherRandomCAProvider,
			},
			expectedConfigMaps: map[string]*corev1.ConfigMap{
				"extension-apiserver-authentication": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"requestheader-username-headers":     `[]`,
						"requestheader-group-headers":        `[]`,
						"requestheader-extra-headers-prefix": `[]`,
						"requestheader-client-ca-file":       string(anotherRandomCA),
						"requestheader-allowed-names":        `[]`,
					},
				},
			},
			expectCreate: true,
		},
		{
			name: "overwrite extension-apiserver-authentication",
			clusterAuthInfo: ClusterAuthenticationInfo{
				ClientCA: someRandomCAProvider,
			},
			preexistingObjs: []runtime.Object{
				&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"client-ca-file": string(anotherRandomCA),
					},
				},
			},
			expectedConfigMaps: map[string]*corev1.ConfigMap{
				"extension-apiserver-authentication": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"client-ca-file": string(anotherRandomCA) + string(someRandomCA),
					},
				},
			},
		},
		{
			name: "overwrite extension-apiserver-authentication requestheader",
			clusterAuthInfo: ClusterAuthenticationInfo{
				RequestHeaderUsernameHeaders:     headerrequest.StaticStringSlice{},
				RequestHeaderGroupHeaders:        headerrequest.StaticStringSlice{},
				RequestHeaderExtraHeaderPrefixes: headerrequest.StaticStringSlice{},
				RequestHeaderCA:                  anotherRandomCAProvider,
				RequestHeaderAllowedNames:        headerrequest.StaticStringSlice{},
			},
			preexistingObjs: []runtime.Object{
				&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"requestheader-username-headers":     `[]`,
						"requestheader-group-headers":        `[]`,
						"requestheader-extra-headers-prefix": `[]`,
						"requestheader-client-ca-file":       string(someRandomCA),
						"requestheader-allowed-names":        `[]`,
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
						"requestheader-client-ca-file":       string(someRandomCA) + string(anotherRandomCA),
						"requestheader-allowed-names":        `[]`,
					},
				},
			},
		},
		{
			name: "namespace exists",
			clusterAuthInfo: ClusterAuthenticationInfo{
				ClientCA: someRandomCAProvider,
			},
			preexistingObjs: []runtime.Object{
				&corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: metav1.NamespaceSystem}},
			},
			expectedConfigMaps: map[string]*corev1.ConfigMap{
				"extension-apiserver-authentication": {
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"client-ca-file": string(someRandomCA),
					},
				},
			},
			expectCreate: true,
		},
		{
			name: "skip on no change",
			clusterAuthInfo: ClusterAuthenticationInfo{
				RequestHeaderUsernameHeaders:     headerrequest.StaticStringSlice{},
				RequestHeaderGroupHeaders:        headerrequest.StaticStringSlice{},
				RequestHeaderExtraHeaderPrefixes: headerrequest.StaticStringSlice{},
				RequestHeaderCA:                  anotherRandomCAProvider,
				RequestHeaderAllowedNames:        headerrequest.StaticStringSlice{},
			},
			preexistingObjs: []runtime.Object{
				&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
					Data: map[string]string{
						"requestheader-username-headers":     `[]`,
						"requestheader-group-headers":        `[]`,
						"requestheader-extra-headers-prefix": `[]`,
						"requestheader-client-ca-file":       string(anotherRandomCA),
						"requestheader-allowed-names":        `[]`,
					},
				},
			},
			expectedConfigMaps: map[string]*corev1.ConfigMap{},
			expectCreate:       false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := fake.NewSimpleClientset(test.preexistingObjs...)
			configMapIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			for _, obj := range test.preexistingObjs {
				configMapIndexer.Add(obj)
			}
			configmapLister := corev1listers.NewConfigMapLister(configMapIndexer)

			c := &Controller{
				configMapLister:            configmapLister,
				configMapClient:            client.CoreV1(),
				namespaceClient:            client.CoreV1(),
				requiredAuthenticationData: test.clusterAuthInfo,
			}

			err := c.syncConfigMap()
			if err != nil {
				t.Fatal(err)
			}

			actualConfigMaps, updated := getFinalConfigMaps(t, client)
			if !reflect.DeepEqual(test.expectedConfigMaps, actualConfigMaps) {
				t.Fatalf("%s: %v", test.name, diff.ObjectReflectDiff(test.expectedConfigMaps, actualConfigMaps))
			}
			if test.expectCreate != updated {
				t.Fatalf("%s: expected %v, got %v", test.name, test.expectCreate, updated)
			}
		})
	}
}

func getFinalConfigMaps(t *testing.T, client *fake.Clientset) (map[string]*corev1.ConfigMap, bool) {
	ret := map[string]*corev1.ConfigMap{}
	created := false

	for _, action := range client.Actions() {
		t.Log(spew.Sdump(action))
		if action.Matches("create", "configmaps") {
			created = true
			obj := action.(clienttesting.CreateAction).GetObject().(*corev1.ConfigMap)
			ret[obj.Name] = obj
		}
		if action.Matches("update", "configmaps") {
			obj := action.(clienttesting.UpdateAction).GetObject().(*corev1.ConfigMap)
			ret[obj.Name] = obj
		}
	}
	return ret, created
}

func TestWriteConfigMapDeleted(t *testing.T) {
	// the basics are tested above, this checks the deletion logic when the ca bundles are too large
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "extension-apiserver-authentication"},
		Data: map[string]string{
			"requestheader-username-headers":     `[]`,
			"requestheader-group-headers":        `[]`,
			"requestheader-extra-headers-prefix": `[]`,
			"requestheader-client-ca-file":       string(anotherRandomCA),
			"requestheader-allowed-names":        `[]`,
		},
	}

	t.Run("request entity too large", func(t *testing.T) {
		client := fake.NewSimpleClientset()
		client.PrependReactor("update", "configmaps", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
			return true, nil, apierrors.NewRequestEntityTooLargeError("way too big")
		})
		client.PrependReactor("delete", "configmaps", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
			return true, nil, nil
		})

		err := writeConfigMap(client.CoreV1(), cm)
		if err == nil || err.Error() != "Request entity too large: way too big" {
			t.Fatal(err)
		}
		if len(client.Actions()) != 2 {
			t.Fatal(client.Actions())
		}
		_, ok := client.Actions()[1].(clienttesting.DeleteAction)
		if !ok {
			t.Fatal(client.Actions())
		}
	})

	t.Run("ca bundle too large", func(t *testing.T) {
		client := fake.NewSimpleClientset()
		client.PrependReactor("update", "configmaps", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
			return true, nil, apierrors.NewInvalid(schema.GroupKind{Kind: "ConfigMap"}, cm.Name, field.ErrorList{field.TooLong(field.NewPath(""), cm, corev1.MaxSecretSize)})
		})
		client.PrependReactor("delete", "configmaps", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
			return true, nil, nil
		})

		err := writeConfigMap(client.CoreV1(), cm)
		if err == nil || err.Error() != `ConfigMap "extension-apiserver-authentication" is invalid: []: Too long: must have at most 1048576 bytes` {
			t.Fatal(err)
		}
		if len(client.Actions()) != 2 {
			t.Fatal(client.Actions())
		}
		_, ok := client.Actions()[1].(clienttesting.DeleteAction)
		if !ok {
			t.Fatal(client.Actions())
		}
	})

}
