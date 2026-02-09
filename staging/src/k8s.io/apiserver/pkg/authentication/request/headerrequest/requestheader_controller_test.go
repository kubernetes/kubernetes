/*
Copyright 2020 The Kubernetes Authors.

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

package headerrequest

import (
	"context"
	"encoding/json"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
)

const (
	defConfigMapName      = "extension-apiserver-authentication"
	defConfigMapNamespace = "kube-system"

	defUsernameHeadersKey     = "user-key"
	defUIDHeadersKey          = "uid-key"
	defGroupHeadersKey        = "group-key"
	defExtraHeaderPrefixesKey = "extra-key"
	defAllowedClientNamesKey  = "names-key"
)

type expectedHeadersHolder struct {
	usernameHeaders     []string
	uidHeaders          []string
	groupHeaders        []string
	extraHeaderPrefixes []string
	allowedClientNames  []string
}

func TestRequestHeaderAuthRequestController(t *testing.T) {
	scenarios := []struct {
		name           string
		cm             *corev1.ConfigMap
		expectedHeader expectedHeadersHolder
		expectErr      bool
	}{
		{
			name: "happy-path: headers values are populated form a config map",
			cm:   defaultConfigMap(t, []string{"user-val"}, []string{"uid-val"}, []string{"group-val"}, []string{"extra-val"}, []string{"names-val"}),
			expectedHeader: expectedHeadersHolder{
				usernameHeaders:     []string{"user-val"},
				uidHeaders:          []string{"uid-val"},
				groupHeaders:        []string{"group-val"},
				extraHeaderPrefixes: []string{"extra-val"},
				allowedClientNames:  []string{"names-val"},
			},
		},
		{
			name: "passing an empty config map doesn't break the controller",
			cm: func() *corev1.ConfigMap {
				c := defaultConfigMap(t, nil, nil, nil, nil, nil)
				c.Data = map[string]string{}
				return c
			}(),
		},
		{
			name: "an invalid config map produces an error",
			cm: func() *corev1.ConfigMap {
				c := defaultConfigMap(t, nil, nil, nil, nil, nil)
				c.Data = map[string]string{
					defUsernameHeadersKey: "incorrect-json-array",
				}
				return c
			}(),
			expectErr: true,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			// test data
			indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
			if err := indexer.Add(scenario.cm); err != nil {
				t.Fatal(err.Error())
			}
			target := newDefaultTarget()
			target.configmapLister = corev1listers.NewConfigMapLister(indexer).ConfigMaps(defConfigMapNamespace)

			// act
			err := target.sync()

			if err != nil && !scenario.expectErr {
				t.Errorf("got unexpected error %v", err)
			}
			if err == nil && scenario.expectErr {
				t.Error("expected an error but didn't get one")
			}

			// validate
			validateExpectedHeaders(t, target, scenario.expectedHeader)
		})
	}
}

func TestRequestHeaderAuthRequestControllerPreserveState(t *testing.T) {
	scenarios := []struct {
		name           string
		cm             *corev1.ConfigMap
		expectedHeader expectedHeadersHolder
		expectErr      bool
	}{
		{
			name: "scenario 1: headers values are populated form a config map",
			cm:   defaultConfigMap(t, []string{"user-val"}, []string{"uid-val"}, []string{"group-val"}, []string{"extra-val"}, []string{"names-val"}),
			expectedHeader: expectedHeadersHolder{
				usernameHeaders:     []string{"user-val"},
				uidHeaders:          []string{"uid-val"},
				groupHeaders:        []string{"group-val"},
				extraHeaderPrefixes: []string{"extra-val"},
				allowedClientNames:  []string{"names-val"},
			},
		},
		{
			name: "scenario 2: an invalid config map produces an error but doesn't destroy the state (scenario 1)",
			cm: func() *corev1.ConfigMap {
				c := defaultConfigMap(t, nil, nil, nil, nil, nil)
				c.Data = map[string]string{
					defUsernameHeadersKey: "incorrect-json-array",
				}
				return c
			}(),
			expectErr: true,
			expectedHeader: expectedHeadersHolder{
				usernameHeaders:     []string{"user-val"},
				uidHeaders:          []string{"uid-val"},
				groupHeaders:        []string{"group-val"},
				extraHeaderPrefixes: []string{"extra-val"},
				allowedClientNames:  []string{"names-val"},
			},
		},
		{
			name: "scenario 3: some headers values have changed (prev set by scenario 1)",
			cm:   defaultConfigMap(t, []string{"user-val"}, []string{"uid-val"}, []string{"group-val-scenario-3"}, []string{"extra-val"}, []string{"names-val"}),
			expectedHeader: expectedHeadersHolder{
				usernameHeaders:     []string{"user-val"},
				uidHeaders:          []string{"uid-val"},
				groupHeaders:        []string{"group-val-scenario-3"},
				extraHeaderPrefixes: []string{"extra-val"},
				allowedClientNames:  []string{"names-val"},
			},
		},
		{
			name: "scenario 4: all headers values have changed (prev set by scenario 3)",
			cm:   defaultConfigMap(t, []string{"user-val-scenario-4"}, []string{"uid-val-scenario-4"}, []string{"group-val-scenario-4"}, []string{"extra-val-scenario-4"}, []string{"names-val-scenario-4"}),
			expectedHeader: expectedHeadersHolder{
				usernameHeaders:     []string{"user-val-scenario-4"},
				uidHeaders:          []string{"uid-val-scenario-4"},
				groupHeaders:        []string{"group-val-scenario-4"},
				extraHeaderPrefixes: []string{"extra-val-scenario-4"},
				allowedClientNames:  []string{"names-val-scenario-4"},
			},
		},
	}

	target := newDefaultTarget()

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			// test data
			if scenario.cm != nil {
				indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
				if err := indexer.Add(scenario.cm); err != nil {
					t.Fatal(err.Error())
				}
				target.configmapLister = corev1listers.NewConfigMapLister(indexer).ConfigMaps(defConfigMapNamespace)
			}

			// act
			err := target.sync()

			if err != nil && !scenario.expectErr {
				t.Errorf("got unexpected error %v", err)
			}
			if err == nil && scenario.expectErr {
				t.Error("expected an error but didn't get one")
			}

			// validate
			validateExpectedHeaders(t, target, scenario.expectedHeader)
		})
	}
}

func TestRequestHeaderAuthRequestControllerSyncOnce(t *testing.T) {
	scenarios := []struct {
		name           string
		cm             *corev1.ConfigMap
		expectedHeader expectedHeadersHolder
		expectErr      bool
	}{
		{
			name: "headers values are populated form a config map",
			cm:   defaultConfigMap(t, []string{"user-val"}, []string{"uid-val"}, []string{"group-val"}, []string{"extra-val"}, []string{"names-val"}),
			expectedHeader: expectedHeadersHolder{
				usernameHeaders:     []string{"user-val"},
				uidHeaders:          []string{"uid-val"},
				groupHeaders:        []string{"group-val"},
				extraHeaderPrefixes: []string{"extra-val"},
				allowedClientNames:  []string{"names-val"},
			},
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			// test data
			target := newDefaultTarget()
			fakeKubeClient := fake.NewSimpleClientset(scenario.cm)
			target.client = fakeKubeClient

			// act
			ctx := context.TODO()
			err := target.RunOnce(ctx)

			if err != nil && !scenario.expectErr {
				t.Errorf("got unexpected error %v", err)
			}
			if err == nil && scenario.expectErr {
				t.Error("expected an error but didn't get one")
			}

			// validate
			validateExpectedHeaders(t, target, scenario.expectedHeader)
		})
	}
}

func defaultConfigMap(t *testing.T, usernameHeaderVal, uidHeaderVal, groupHeadersVal, extraHeaderPrefixesVal, allowedClientNamesVal []string) *corev1.ConfigMap {
	encode := func(val []string) string {
		encodedVal, err := json.Marshal(val)
		if err != nil {
			t.Fatalf("unable to marshal %q , due to %v", usernameHeaderVal, err)
		}
		return string(encodedVal)
	}
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      defConfigMapName,
			Namespace: defConfigMapNamespace,
		},
		Data: map[string]string{
			defUsernameHeadersKey:     encode(usernameHeaderVal),
			defUIDHeadersKey:          encode(uidHeaderVal),
			defGroupHeadersKey:        encode(groupHeadersVal),
			defExtraHeaderPrefixesKey: encode(extraHeaderPrefixesVal),
			defAllowedClientNamesKey:  encode(allowedClientNamesVal),
		},
	}
}

func newDefaultTarget() *RequestHeaderAuthRequestController {
	return &RequestHeaderAuthRequestController{
		configmapName:          defConfigMapName,
		configmapNamespace:     defConfigMapNamespace,
		usernameHeadersKey:     defUsernameHeadersKey,
		uidHeadersKey:          defUIDHeadersKey,
		groupHeadersKey:        defGroupHeadersKey,
		extraHeaderPrefixesKey: defExtraHeaderPrefixesKey,
		allowedClientNamesKey:  defAllowedClientNamesKey,
	}
}

func validateExpectedHeaders(t *testing.T, target *RequestHeaderAuthRequestController, expected expectedHeadersHolder) {
	if !equality.Semantic.DeepEqual(target.UsernameHeaders(), expected.usernameHeaders) {
		t.Fatalf("incorrect usernameHeaders, got %v, wanted %v", target.UsernameHeaders(), expected.usernameHeaders)
	}
	if !equality.Semantic.DeepEqual(target.GroupHeaders(), expected.groupHeaders) {
		t.Fatalf("incorrect groupHeaders, got %v, wanted %v", target.GroupHeaders(), expected.groupHeaders)
	}
	if !equality.Semantic.DeepEqual(target.ExtraHeaderPrefixes(), expected.extraHeaderPrefixes) {
		t.Fatalf("incorrect extraheaderPrefixes, got %v, wanted %v", target.ExtraHeaderPrefixes(), expected.extraHeaderPrefixes)
	}
	if !equality.Semantic.DeepEqual(target.AllowedClientNames(), expected.allowedClientNames) {
		t.Fatalf("incorrect expectedAllowedClientNames, got %v, wanted %v", target.AllowedClientNames(), expected.allowedClientNames)
	}
}
