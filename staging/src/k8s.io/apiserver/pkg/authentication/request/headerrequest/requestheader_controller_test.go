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
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"errors"
	"math/big"
	"net/http"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	x509request "k8s.io/apiserver/pkg/authentication/request/x509"
	"k8s.io/client-go/kubernetes/fake"
	corev1listers "k8s.io/client-go/listers/core/v1"
	clienttesting "k8s.io/client-go/testing"
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

// expectedHeadersAfterDeletion is the state both sync and RunOnce must reach once the
// configmap is gone: every header name dropped, but the last allowed client names kept.
// An empty allowed-names list means "allow any common name" to x509.Verifier, so dropping
// them too would widen the common name check instead of revoking trust.
var expectedHeadersAfterDeletion = expectedHeadersHolder{allowedClientNames: []string{"names-val"}}

func TestRequestHeaderAuthRequestControllerSyncClearsHeadersOnConfigMapDeletion(t *testing.T) {
	target := newDefaultTarget()

	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	if err := indexer.Add(defaultConfigMap(t, []string{"user-val"}, []string{"uid-val"}, []string{"group-val"}, []string{"extra-val"}, []string{"names-val"})); err != nil {
		t.Fatal(err)
	}
	target.configmapLister = corev1listers.NewConfigMapLister(indexer).ConfigMaps(defConfigMapNamespace)
	if err := target.sync(); err != nil {
		t.Fatal(err)
	}

	// the configmap is deleted, so the lister no longer has it
	target.configmapLister = corev1listers.NewConfigMapLister(cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})).ConfigMaps(defConfigMapNamespace)
	if err := target.sync(); err != nil {
		t.Fatalf("sync should swallow the not found error, got %v", err)
	}

	validateExpectedHeaders(t, target, expectedHeadersAfterDeletion)
}

func TestRequestHeaderAuthRequestControllerSyncKeepsHeadersOnMalformedConfigMap(t *testing.T) {
	target := newDefaultTarget()

	expected := expectedHeadersHolder{
		usernameHeaders:     []string{"user-val"},
		uidHeaders:          []string{"uid-val"},
		groupHeaders:        []string{"group-val"},
		extraHeaderPrefixes: []string{"extra-val"},
		allowedClientNames:  []string{"names-val"},
	}
	cm := defaultConfigMap(t, expected.usernameHeaders, expected.uidHeaders, expected.groupHeaders, expected.extraHeaderPrefixes, expected.allowedClientNames)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	if err := indexer.Add(cm); err != nil {
		t.Fatal(err)
	}
	target.configmapLister = corev1listers.NewConfigMapLister(indexer).ConfigMaps(defConfigMapNamespace)
	if err := target.sync(); err != nil {
		t.Fatal(err)
	}

	// a malformed configmap is not a deletion, so the last good bundle has to survive
	broken := cm.DeepCopy()
	broken.Data[defUsernameHeadersKey] = "not-a-json-array"
	brokenIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	if err := brokenIndexer.Add(broken); err != nil {
		t.Fatal(err)
	}
	target.configmapLister = corev1listers.NewConfigMapLister(brokenIndexer).ConfigMaps(defConfigMapNamespace)
	if err := target.sync(); err == nil {
		t.Fatal("expected sync to report the malformed configmap")
	}

	validateExpectedHeaders(t, target, expected)
}

func TestRequestHeaderAuthRequestControllerRunOnceClearsHeadersOnConfigMapDeletion(t *testing.T) {
	target := newDefaultTarget()
	if err := target.syncConfigMap(defaultConfigMap(t, []string{"user-val"}, []string{"uid-val"}, []string{"group-val"}, []string{"extra-val"}, []string{"names-val"})); err != nil {
		t.Fatal(err)
	}

	target.client = fake.NewSimpleClientset()
	if err := target.RunOnce(context.TODO()); err != nil {
		t.Fatal(err)
	}

	validateExpectedHeaders(t, target, expectedHeadersAfterDeletion)
}

func TestRequestHeaderAuthRequestControllerRunOnceKeepsHeadersOnOtherAPIErrors(t *testing.T) {
	target := newDefaultTarget()
	expected := expectedHeadersHolder{
		usernameHeaders:     []string{"user-val"},
		uidHeaders:          []string{"uid-val"},
		groupHeaders:        []string{"group-val"},
		extraHeaderPrefixes: []string{"extra-val"},
		allowedClientNames:  []string{"names-val"},
	}
	if err := target.syncConfigMap(defaultConfigMap(t, expected.usernameHeaders, expected.uidHeaders, expected.groupHeaders, expected.extraHeaderPrefixes, expected.allowedClientNames)); err != nil {
		t.Fatal(err)
	}

	client := fake.NewSimpleClientset()
	client.PrependReactor("get", "configmaps", func(clienttesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.New("temporary API error")
	})
	target.client = client

	if err := target.RunOnce(context.TODO()); err == nil {
		t.Fatal("expected RunOnce to return the API error")
	}

	validateExpectedHeaders(t, target, expected)
}

// TestRequestHeaderAuthRequestControllerClearedBundleRejectsRecreatedConfigMapRace covers the
// window where the configmap is deleted and immediately recreated, which is what happens in a
// real cluster because the kube-apiserver rewrites it. The x509 verifier reads the allowed
// client names before the wrapped handler reads the username headers, so those two reads can
// land on either side of the recreation. If the cleared bundle dropped the allowed client
// names, this read ordering would skip the common name check entirely and then honor the
// restored username headers, letting a certificate through that the configmap never allowed.
func TestRequestHeaderAuthRequestControllerClearedBundleRejectsRecreatedConfigMapRace(t *testing.T) {
	target := newDefaultTarget()

	// the client certificate generated below uses the common name "not-allowed-client",
	// which is deliberately absent from the configmap's allowed list ("front-proxy-client").
	configMap := defaultConfigMap(t, []string{"X-Remote-User"}, nil, nil, nil, []string{"front-proxy-client"})
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	if err := indexer.Add(configMap); err != nil {
		t.Fatal(err)
	}
	target.configmapLister = corev1listers.NewConfigMapLister(indexer).ConfigMaps(defConfigMapNamespace)
	if err := target.sync(); err != nil {
		t.Fatal(err)
	}

	// the configmap is deleted
	target.configmapLister = corev1listers.NewConfigMapLister(cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})).ConfigMaps(defConfigMapNamespace)
	if err := target.sync(); err != nil {
		t.Fatal(err)
	}

	// it comes back, but the controller has not synced it yet
	recreatedIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	if err := recreatedIndexer.Add(configMap); err != nil {
		t.Fatal(err)
	}
	target.configmapLister = corev1listers.NewConfigMapLister(recreatedIndexer).ConfigMaps(defConfigMapNamespace)

	roots, clientCert := newTestClientCert(t, "not-allowed-client")

	usernameHeadersRead := false
	// The CA controller is independent and may still be serving the old CA, so keep
	// certificate verification passing and let the common name check decide.
	auth := NewDynamicVerifyOptionsSecure(
		x509request.StaticVerifierFn(x509.VerifyOptions{
			Roots:     roots,
			KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		}),
		StringSliceProviderFunc(target.AllowedClientNames),
		StringSliceProviderFunc(func() []string {
			// the recreation becomes visible only after the common name check has run
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
	req.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{clientCert}}

	resp, ok, err := auth.AuthenticateRequest(req)
	if err == nil {
		t.Fatal("expected the preserved allowed client names to reject the certificate")
	}
	if ok {
		t.Fatalf("unexpected successful authentication as %v", resp)
	}
	if usernameHeadersRead {
		t.Fatal("username headers were read before the common name check, so this test no longer simulates the race it was written for; rebuild the simulation rather than deleting this assertion")
	}
}

// newTestClientCert returns a root pool and a client certificate signed by that root,
// so the certificate chain verifies and the common name check is what decides the request.
func newTestClientCert(t *testing.T, commonName string) (*x509.CertPool, *x509.Certificate) {
	t.Helper()

	notBefore := time.Now().Add(-time.Hour)
	notAfter := time.Now().Add(time.Hour)

	caKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	caTemplate := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "test-requestheader-ca"},
		NotBefore:             notBefore,
		NotAfter:              notAfter,
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageCertSign,
	}
	caDER, err := x509.CreateCertificate(rand.Reader, caTemplate, caTemplate, &caKey.PublicKey, caKey)
	if err != nil {
		t.Fatal(err)
	}
	caCert, err := x509.ParseCertificate(caDER)
	if err != nil {
		t.Fatal(err)
	}

	clientKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	clientTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject:      pkix.Name{CommonName: commonName},
		NotBefore:    notBefore,
		NotAfter:     notAfter,
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	clientDER, err := x509.CreateCertificate(rand.Reader, clientTemplate, caCert, &clientKey.PublicKey, caKey)
	if err != nil {
		t.Fatal(err)
	}
	clientCert, err := x509.ParseCertificate(clientDER)
	if err != nil {
		t.Fatal(err)
	}

	roots := x509.NewCertPool()
	roots.AddCert(caCert)
	return roots, clientCert
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
	if !equality.Semantic.DeepEqual(target.UIDHeaders(), expected.uidHeaders) {
		t.Fatalf("incorrect uidHeaders, got %v, wanted %v", target.UIDHeaders(), expected.uidHeaders)
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
