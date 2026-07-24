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
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	x509request "k8s.io/apiserver/pkg/authentication/request/x509"
	"k8s.io/client-go/kubernetes/fake"
	corev1listers "k8s.io/client-go/listers/core/v1"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	certutil "k8s.io/client-go/util/cert"
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
	// allowedClientNames may be nil or empty to allow all common names when
	// rejectAllClientNames is false.
	allowedClientNames []string
	// rejectAllClientNames represents deleted or unavailable ConfigMap state and
	// rejects all common names, even when allowedClientNames is nil or empty.
	rejectAllClientNames bool
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
			expectedHeader: expectedHeadersHolder{rejectAllClientNames: true},
			expectErr:      true,
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

func TestRequestHeaderAuthRequestControllerRunOnceClearsStateOnConfigMapDeletion(t *testing.T) {
	target := newDefaultTarget()
	if err := target.syncConfigMap(defaultConfigMap(t, []string{"user-val"}, []string{"uid-val"}, []string{"group-val"}, []string{"extra-val"}, []string{"names-val"})); err != nil {
		t.Fatal(err)
	}
	target.client = fake.NewSimpleClientset()

	if err := target.RunOnce(context.Background()); err != nil {
		t.Fatal(err)
	}

	validateExpectedHeaders(t, target, expectedHeadersHolder{
		rejectAllClientNames: true,
	})
}

func TestRequestHeaderAuthRequestControllerRunOncePreservesStateOnNonNotFoundError(t *testing.T) {
	target := newDefaultTarget()
	expectedHeaders := expectedHeadersHolder{
		usernameHeaders:     []string{"user-val"},
		uidHeaders:          []string{"uid-val"},
		groupHeaders:        []string{"group-val"},
		extraHeaderPrefixes: []string{"extra-val"},
		allowedClientNames:  []string{"names-val"},
	}
	if err := target.syncConfigMap(defaultConfigMap(
		t,
		expectedHeaders.usernameHeaders,
		expectedHeaders.uidHeaders,
		expectedHeaders.groupHeaders,
		expectedHeaders.extraHeaderPrefixes,
		expectedHeaders.allowedClientNames,
	)); err != nil {
		t.Fatal(err)
	}

	client := fake.NewSimpleClientset()
	client.PrependReactor("get", "configmaps", func(clienttesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.New("temporary API error")
	})
	target.client = client

	if err := target.RunOnce(context.Background()); err == nil {
		t.Fatal("expected RunOnce to return the API error")
	}

	validateExpectedHeaders(t, target, expectedHeaders)
}

func TestRequestHeaderAuthRequestControllerDeletedBundleRejectsRequestDuringConfigMapRecreation(t *testing.T) {
	target := newDefaultTarget()

	configMap := defaultConfigMap(t, []string{"X-Remote-User"}, nil, nil, nil, []string{"front-proxy-client"})
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	if err := indexer.Add(configMap); err != nil {
		t.Fatal(err)
	}
	target.configmapLister = corev1listers.NewConfigMapLister(indexer).ConfigMaps(defConfigMapNamespace)
	if err := target.sync(); err != nil {
		t.Fatal(err)
	}

	// Delete the ConfigMap, then make the recreated ConfigMap visible without
	// syncing it. The username header provider below will observe the recreation
	// only after the x509 verifier has read the deleted state.
	target.configmapLister = corev1listers.NewConfigMapLister(cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})).ConfigMaps(defConfigMapNamespace)
	if err := target.sync(); err != nil {
		t.Fatal(err)
	}

	recreatedIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	if err := recreatedIndexer.Add(configMap); err != nil {
		t.Fatal(err)
	}
	target.configmapLister = corev1listers.NewConfigMapLister(recreatedIndexer).ConfigMaps(defConfigMapNamespace)

	// Reuse the x509 package's multi-level ClientAuth fixture to match the
	// ConfigMap CA controller's client-auth verification.
	clientCerts, err := certutil.CertsFromFile("../x509/testdata/client-valid.pem")
	if err != nil {
		t.Fatal(err)
	}
	intermediateCerts, err := certutil.CertsFromFile("../x509/testdata/intermediate.pem")
	if err != nil {
		t.Fatal(err)
	}
	roots, err := certutil.NewPool("../x509/testdata/root.pem")
	if err != nil {
		t.Fatal(err)
	}

	usernameHeadersRead := false
	// Keep CA verification valid to model the independently updated CA controller
	// still exposing the same CA during the request-header ConfigMap transition.
	auth := NewDynamicVerifyOptionsSecure(
		x509request.StaticVerifierFn(x509.VerifyOptions{
			Roots:     roots,
			KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		}),
		target.AllowedClientNamesFunc(),
		StringSliceProviderFunc(func() []string {
			usernameHeadersRead = true
			if err := target.sync(); err != nil {
				t.Fatal(err)
			}
			return target.UsernameHeaders()
		}),
		StringSliceProviderFunc(target.UIDHeaders),
		StringSliceProviderFunc(target.GroupHeaders),
		StringSliceProviderFunc(target.ExtraHeaderPrefixes),
	)

	req, err := http.NewRequest(http.MethodGet, "/", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("X-Remote-User", "spoofed-user")
	req.TLS = &tls.ConnectionState{PeerCertificates: append(clientCerts, intermediateCerts...)}

	resp, ok, err := auth.AuthenticateRequest(req)
	if err == nil {
		t.Fatal("expected the deleted allowed client names to reject the certificate")
	}
	if ok {
		t.Fatal("unexpected successful authentication")
	}
	if resp != nil {
		t.Fatalf("unexpected authentication response: %#v", resp)
	}
	if usernameHeadersRead {
		t.Fatal("username headers were read after the deleted allowed client names were checked")
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
	target := &RequestHeaderAuthRequestController{
		configmapName:          defConfigMapName,
		configmapNamespace:     defConfigMapNamespace,
		usernameHeadersKey:     defUsernameHeadersKey,
		uidHeadersKey:          defUIDHeadersKey,
		groupHeadersKey:        defGroupHeadersKey,
		extraHeaderPrefixesKey: defExtraHeaderPrefixesKey,
		allowedClientNamesKey:  defAllowedClientNamesKey,
	}
	target.clearRequestHeaderBundle()
	return target
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
	_, rejectAllClientNames := target.AllowedClientNamesFunc()()
	if rejectAllClientNames != expected.rejectAllClientNames {
		t.Fatalf("incorrect rejectAllClientNames, got %t, wanted %t", rejectAllClientNames, expected.rejectAllClientNames)
	}
}
