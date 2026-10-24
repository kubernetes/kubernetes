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

package headerrequest

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"math/big"
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apiserver/pkg/authentication/user"
)

func TestRequestHeader(t *testing.T) {
	testcases := map[string]struct {
		nameHeaders        []string
		uidHeaders         []string
		groupHeaders       []string
		extraPrefixHeaders []string
		requestHeaders     http.Header
		finalHeaders       http.Header

		expectedUser user.Info
		expectedOk   bool
	}{
		"empty": {},
		"user no match": {
			nameHeaders: []string{"X-Remote-User"},
		},
		"user match": {
			nameHeaders:    []string{"X-Remote-User"},
			requestHeaders: http.Header{"X-Remote-User": {"Bob"}},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				Groups: []string{},
				Extra:  map[string][]string{},
			},
			expectedOk: true,
		},
		"user exact match": {
			nameHeaders: []string{"X-Remote-User"},
			requestHeaders: http.Header{
				"Prefixed-X-Remote-User-With-Suffix": {"Bob"},
				"X-Remote-User-With-Suffix":          {"Bob"},
			},
		},
		"user first match": {
			nameHeaders: []string{
				"X-Remote-User",
				"A-Second-X-Remote-User",
				"Another-X-Remote-User",
			},
			requestHeaders: http.Header{
				"X-Remote-User":          {"", "First header, second value"},
				"A-Second-X-Remote-User": {"Second header, first value", "Second header, second value"},
				"Another-X-Remote-User":  {"Third header, first value"}},
			expectedUser: &user.DefaultInfo{
				Name:   "Second header, first value",
				Groups: []string{},
				Extra:  map[string][]string{},
			},
			expectedOk: true,
		},
		"user case-insensitive": {
			nameHeaders:    []string{"x-REMOTE-user"},             // configured headers can be case-insensitive
			requestHeaders: http.Header{"X-Remote-User": {"Bob"}}, // the parsed headers are normalized by the http package
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				Groups: []string{},
				Extra:  map[string][]string{},
			},
			expectedOk: true,
		},

		"groups none": {
			nameHeaders:  []string{"X-Remote-User"},
			groupHeaders: []string{"X-Remote-Group"},
			requestHeaders: http.Header{
				"X-Remote-User": {"Bob"},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				Groups: []string{},
				Extra:  map[string][]string{},
			},
			expectedOk: true,
		},
		"groups all matches": {
			nameHeaders:  []string{"X-Remote-User"},
			groupHeaders: []string{"X-Remote-Group-1", "X-Remote-Group-2"},
			requestHeaders: http.Header{
				"X-Remote-User":    {"Bob"},
				"X-Remote-Group-1": {"one-a", "one-b"},
				"X-Remote-Group-2": {"two-a", "two-b"},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				Groups: []string{"one-a", "one-b", "two-a", "two-b"},
				Extra:  map[string][]string{},
			},
			expectedOk: true,
		},
		"groups case-insensitive": {
			nameHeaders:  []string{"X-REMOTE-User"},
			groupHeaders: []string{"X-REMOTE-Group"},
			requestHeaders: http.Header{
				"X-Remote-User":  {"Bob"},
				"X-Remote-Group": {"Users"},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				Groups: []string{"Users"},
				Extra:  map[string][]string{},
			},
			expectedOk: true,
		},
		"uid none": {
			nameHeaders: []string{"X-Remote-User"},
			uidHeaders:  []string{"X-Remote-Uid"},
			requestHeaders: http.Header{
				"X-Remote-User": {"Bob"},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				UID:    "",
				Groups: []string{},
				Extra:  map[string][]string{},
			},
			expectedOk: true,
		},
		"uid exact match": {
			nameHeaders: []string{"X-Remote-User"},
			uidHeaders:  []string{"X-Remote-Uid"},
			requestHeaders: http.Header{
				"X-Remote-User": {"Bob"},
				// The keys in http.Header MUST be http.CanonicalHeaderKey.
				// Hence X-Remote-Uid-1 instead of X-Remote-UID-1.
				"X-Remote-Uid-1": {"8f5ea9d1-a5ed-4d02-80a2-26709216350b"},
				"X-Remote-Uid-2": {"c7644180-c774-4a9b-81e5-3eef76f087ab"},
			},
			finalHeaders: http.Header{
				"X-Remote-Uid-1": {"8f5ea9d1-a5ed-4d02-80a2-26709216350b"},
				"X-Remote-Uid-2": {"c7644180-c774-4a9b-81e5-3eef76f087ab"},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				UID:    "",
				Groups: []string{},
				Extra:  map[string][]string{},
			},
			expectedOk: true,
		},
		"uid first match": {
			nameHeaders: []string{"X-Remote-User"},
			uidHeaders:  []string{"X-Remote-Uid-1", "X-Remote-Uid-2"},
			requestHeaders: http.Header{
				"X-Remote-User":  {"Bob"},
				"X-Remote-Uid-1": {"8f5ea9d1-a5ed-4d02-80a2-26709216350b"},
				"X-Remote-Uid-2": {"c7644180-c774-4a9b-81e5-3eef76f087ab"},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				UID:    "8f5ea9d1-a5ed-4d02-80a2-26709216350b",
				Groups: []string{},
				Extra:  map[string][]string{},
			},
			expectedOk: true,
		},
		"extra prefix matches case-insensitive": {
			nameHeaders:        []string{"X-Remote-User"},
			uidHeaders:         []string{"X-Remote-UID"},
			groupHeaders:       []string{"X-Remote-Group-1", "X-Remote-Group-2"},
			extraPrefixHeaders: []string{"X-Remote-Extra-1-", "X-Remote-Extra-2-"},
			requestHeaders: http.Header{
				"X-Remote-User":         {"Bob"},
				"X-Remote-Uid":          {"2ca80fb0-60ea-4ecf-951c-89af843b0402"},
				"X-Remote-Group-1":      {"one-a", "one-b"},
				"X-Remote-Group-2":      {"two-a", "two-b"},
				"X-Remote-extra-1-key1": {"alfa", "bravo"},
				"X-Remote-Extra-1-Key2": {"charlie", "delta"},
				"X-Remote-Extra-1-":     {"india", "juliet"},
				"X-Remote-extra-2-":     {"kilo", "lima"},
				"X-Remote-extra-2-Key1": {"echo", "foxtrot"},
				"X-Remote-Extra-2-key2": {"golf", "hotel"},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				UID:    "2ca80fb0-60ea-4ecf-951c-89af843b0402",
				Groups: []string{"one-a", "one-b", "two-a", "two-b"},
				Extra: map[string][]string{
					"key1": {"alfa", "bravo", "echo", "foxtrot"},
					"key2": {"charlie", "delta", "golf", "hotel"},
					"":     {"india", "juliet", "kilo", "lima"},
				},
			},
			expectedOk: true,
		},

		"extra prefix matches case-insensitive with unrelated headers": {
			nameHeaders:        []string{"X-Remote-User"},
			groupHeaders:       []string{"X-Remote-Group-1", "X-Remote-Group-2"},
			extraPrefixHeaders: []string{"X-Remote-Extra-1-", "X-Remote-Extra-2-"},
			requestHeaders: http.Header{
				"X-Group-Remote":        {"snorlax"}, // unrelated header
				"X-Group-Bear":          {"panda"},   // another unrelated header
				"X-Remote-User":         {"Bob"},
				"X-Remote-Group-1":      {"one-a", "one-b"},
				"X-Remote-Group-2":      {"two-a", "two-b"},
				"X-Remote-extra-1-key1": {"alfa", "bravo"},
				"X-Remote-Extra-1-Key2": {"charlie", "delta"},
				"X-Remote-Extra-1-":     {"india", "juliet"},
				"X-Remote-extra-2-":     {"kilo", "lima"},
				"X-Remote-extra-2-Key1": {"echo", "foxtrot"},
				"X-Remote-Extra-2-key2": {"golf", "hotel"},
			},
			finalHeaders: http.Header{
				"X-Group-Remote": {"snorlax"},
				"X-Group-Bear":   {"panda"},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				Groups: []string{"one-a", "one-b", "two-a", "two-b"},
				Extra: map[string][]string{
					"key1": {"alfa", "bravo", "echo", "foxtrot"},
					"key2": {"charlie", "delta", "golf", "hotel"},
					"":     {"india", "juliet", "kilo", "lima"},
				},
			},
			expectedOk: true,
		},

		"escaped extra keys": {
			nameHeaders:        []string{"X-Remote-User"},
			uidHeaders:         []string{"X-Remote-Uid"},
			groupHeaders:       []string{"X-Remote-Group"},
			extraPrefixHeaders: []string{"X-Remote-Extra-"},
			requestHeaders: http.Header{
				"X-Remote-User":                                            {"Bob"},
				"X-Remote-Uid":                                             {"2ca80fb0-60ea-4ecf-951c-89af843b0402"},
				"X-Remote-Group":                                           {"one-a", "one-b"},
				"X-Remote-Extra-Alpha":                                     {"alphabetical"},
				"X-Remote-Extra-Alph4num3r1c":                              {"alphanumeric"},
				"X-Remote-Extra-Percent%20encoded":                         {"percent encoded"},
				"X-Remote-Extra-Almost%zzpercent%xxencoded":                {"not quite percent encoded"},
				"X-Remote-Extra-Example.com%2fpercent%2520encoded":         {"url with double percent encoding"},
				"X-Remote-Extra-Example.com%2F%E4%BB%8A%E6%97%A5%E3%81%AF": {"url with unicode"},
				"X-Remote-Extra-Abc123!#$+.-_*\\^`~|'":                     {"header key legal characters"},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				UID:    "2ca80fb0-60ea-4ecf-951c-89af843b0402",
				Groups: []string{"one-a", "one-b"},
				Extra: map[string][]string{
					"alpha":                         {"alphabetical"},
					"alph4num3r1c":                  {"alphanumeric"},
					"percent encoded":               {"percent encoded"},
					"almost%zzpercent%xxencoded":    {"not quite percent encoded"},
					"example.com/percent%20encoded": {"url with double percent encoding"},
					"example.com/今日は":               {"url with unicode"},
					"abc123!#$+.-_*\\^`~|'":         {"header key legal characters"},
				},
			},
			expectedOk: true,
		},
	}

	for k, testcase := range testcases {
		t.Run(k, func(t *testing.T) {
			auth, err := New(testcase.nameHeaders, testcase.uidHeaders, testcase.groupHeaders, testcase.extraPrefixHeaders)
			if err != nil {
				t.Fatal(err)
			}
			req := &http.Request{Header: testcase.requestHeaders}

			resp, ok, _ := auth.AuthenticateRequest(req)
			if testcase.expectedOk != ok {
				t.Errorf("%v: expected %v, got %v", k, testcase.expectedOk, ok)
			}
			if !ok {
				return
			}
			if e, a := testcase.expectedUser, resp.User; !reflect.DeepEqual(e, a) {
				t.Errorf("%v: expected %#v, got %#v", k, e, a)
			}

			want := testcase.finalHeaders
			if want == nil && testcase.requestHeaders != nil {
				want = http.Header{}
			}
			if diff := cmp.Diff(want, testcase.requestHeaders); len(diff) > 0 {
				t.Errorf("unexpected final headers (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNewSecure(t *testing.T) {
	roots, clientCerts := newTestClientCertsFromSameCA(t, "front-proxy-client", "other-client")
	allowedClientCert, disallowedClientCert := clientCerts[0], clientCerts[1]

	config := &RequestHeaderConfig{
		UsernameHeaders:     []string{"X-Remote-User"},
		UIDHeaders:          []string{"X-Remote-Uid"},
		GroupHeaders:        []string{"X-Remote-Group"},
		ExtraHeaderPrefixes: []string{"X-Remote-Extra-"},
		AllowedClientNames:  []string{"front-proxy-client"},
	}

	scenarios := []struct {
		name           string
		snapshot       *RequestHeaderConfig
		cert           *x509.Certificate
		requestHeaders http.Header
		expectedUser   user.Info
		expectedOk     bool
		expectErr      bool
	}{
		{
			name:      "nil snapshot fails closed",
			snapshot:  nil,
			cert:      allowedClientCert,
			expectErr: false,
		},
		{
			name:     "empty allowed names accepts any verified certificate",
			snapshot: func() *RequestHeaderConfig { c := *config; c.AllowedClientNames = nil; return &c }(),
			cert:     disallowedClientCert,
			requestHeaders: http.Header{
				"X-Remote-User": {"Bob"},
			},
			expectedUser: &user.DefaultInfo{Name: "Bob", Groups: []string{}, Extra: map[string][]string{}},
			expectedOk:   true,
		},
		{
			name:     "allowed common name is accepted",
			snapshot: config,
			cert:     allowedClientCert,
			requestHeaders: http.Header{
				"X-Remote-User": {"Bob"},
			},
			expectedUser: &user.DefaultInfo{Name: "Bob", Groups: []string{}, Extra: map[string][]string{}},
			expectedOk:   true,
		},
		{
			name:      "common name outside the allowlist is rejected",
			snapshot:  config,
			cert:      disallowedClientCert,
			expectErr: true,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			provider := RequestHeaderConfigProviderFunc(func() *RequestHeaderConfig { return scenario.snapshot })
			auth := NewSecure(provider, &staticVerifyOptionsCA{verifyOptionsFn: func() (x509.VerifyOptions, bool) {
				return x509.VerifyOptions{Roots: roots, KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth}}, true
			}})

			req, err := http.NewRequest(http.MethodGet, "/", nil)
			if err != nil {
				t.Fatal(err)
			}
			req.Header = scenario.requestHeaders
			if req.Header == nil {
				req.Header = http.Header{}
			}
			req.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{scenario.cert}}

			resp, ok, err := auth.AuthenticateRequest(req)
			if scenario.expectErr != (err != nil) {
				t.Fatalf("expected error %v, got %v", scenario.expectErr, err)
			}
			if scenario.expectedOk != ok {
				t.Fatalf("expected ok %v, got %v", scenario.expectedOk, ok)
			}
			if !ok {
				return
			}
			if e, a := scenario.expectedUser, resp.User; !reflect.DeepEqual(e, a) {
				t.Errorf("expected %#v, got %#v", e, a)
			}
		})
	}
}

// TestNewSecureUsesSingleSnapshotPerRequest forces Store(recreatedBundle) after Snapshot()
// returns but before header processing. The request must continue using the original
// snapshot for both CN validation and header extraction. Both client certificates are signed by
// the same CA.
func TestNewSecureUsesSingleSnapshotPerRequest(t *testing.T) {
	roots, clientCerts := newTestClientCertsFromSameCA(t, "front-proxy-client", "attacker")
	legitimateCert, attackerCert := clientCerts[0], clientCerts[1]

	original := &RequestHeaderConfig{
		UsernameHeaders:    []string{"X-Original-User"},
		GroupHeaders:       []string{"X-Original-Group"},
		AllowedClientNames: []string{"front-proxy-client"},
	}
	recreated := &RequestHeaderConfig{
		UsernameHeaders:    []string{"X-New-User"},
		GroupHeaders:       []string{"X-New-Group"},
		AllowedClientNames: nil, // empty means accept any verified client certificate
	}

	target := newDefaultTarget()
	target.exportedRequestHeaderConfig.Store(original)

	var forcedStore bool
	provider := RequestHeaderConfigProviderFunc(func() *RequestHeaderConfig {
		snapshot := target.Snapshot()
		if !forcedStore {
			// simulate the controller loading the recreated configmap between this
			// request's Snapshot() and its header processing
			forcedStore = true
			target.exportedRequestHeaderConfig.Store(recreated)
		}
		return snapshot
	})

	auth := NewSecure(provider, &staticVerifyOptionsCA{verifyOptionsFn: func() (x509.VerifyOptions, bool) {
		return x509.VerifyOptions{Roots: roots, KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth}}, true
	}})

	newRequest := func(cert *x509.Certificate) (*http.Request, error) {
		req, err := http.NewRequest(http.MethodGet, "/", nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("X-New-User", "bob") // only present in the recreated configuration
		req.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{cert}}
		return req, nil
	}

	// The recreated header must not be honored on the first request, even when the client
	// certificate is allowed by the original configuration.
	req, err := newRequest(legitimateCert)
	if err != nil {
		t.Fatal(err)
	}
	resp, ok, err := auth.AuthenticateRequest(req)
	if !forcedStore {
		t.Fatal("expected the forcing provider to run during AuthenticateRequest")
	}
	if err != nil || ok {
		t.Fatalf("expected the request to reject the recreated username header using the original snapshot, got resp=%v ok=%v err=%v", resp, ok, err)
	}

	// Repeat the forced interleaving to verify that the original allowlist also remains in
	// effect for the request that observed it.
	target.exportedRequestHeaderConfig.Store(original)
	forcedStore = false
	req, err = newRequest(attackerCert)
	if err != nil {
		t.Fatal(err)
	}
	resp, ok, err = auth.AuthenticateRequest(req)
	if !forcedStore {
		t.Fatal("expected the forcing provider to run during AuthenticateRequest")
	}
	if err == nil || ok {
		t.Fatalf("expected the request to reject the attacker using the original snapshot's allowlist, got resp=%v ok=%v err=%v", resp, ok, err)
	}

	// a subsequent request observes the recreated configuration and authenticates against it
	req, err = newRequest(legitimateCert)
	if err != nil {
		t.Fatal(err)
	}
	resp, ok, err = auth.AuthenticateRequest(req)
	if err != nil || !ok {
		t.Fatalf("expected authentication to succeed against the recreated configuration, got resp=%v ok=%v err=%v", resp, ok, err)
	}
	if e, a := "bob", resp.User.GetName(); e != a {
		t.Errorf("expected user name %q, got %q", e, a)
	}
}

func TestNewSecureUsesRequestContextSnapshot(t *testing.T) {
	roots, clientCerts := newTestClientCertsFromSameCA(t, "front-proxy-client")
	snapshot := &RequestHeaderConfig{
		UsernameHeaders:    []string{"X-Remote-User"},
		AllowedClientNames: []string{"front-proxy-client"},
	}
	providerCalls := 0
	auth := NewSecure(RequestHeaderConfigProviderFunc(func() *RequestHeaderConfig {
		providerCalls++
		return nil
	}), &staticVerifyOptionsCA{verifyOptionsFn: func() (x509.VerifyOptions, bool) {
		return x509.VerifyOptions{Roots: roots, KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth}}, true
	}})

	req, err := http.NewRequest(http.MethodGet, "/", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("X-Remote-User", "bob")
	req.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{clientCerts[0]}}
	req = req.WithContext(WithRequestHeaderConfig(req.Context(), snapshot))

	resp, ok, err := auth.AuthenticateRequest(req)
	if err != nil || !ok || resp.User.GetName() != "bob" {
		t.Fatalf("expected authentication using the context snapshot, got resp=%v ok=%v err=%v", resp, ok, err)
	}
	if providerCalls != 0 {
		t.Fatalf("expected NewSecure to use the context snapshot without loading the provider, got %d calls", providerCalls)
	}
}

func TestClearAuthenticationHeadersFromConfig(t *testing.T) {
	h := http.Header{
		"X-Original-User":      {"bob"},
		"X-Original-Extra-Foo": {"value"},
		"X-Unrelated":          {"kept"},
	}

	ClearAuthenticationHeadersFromConfig(h, nil)
	if len(h) != 3 {
		t.Fatalf("expected a nil snapshot to be a no-op, got %v", h)
	}

	ClearAuthenticationHeadersFromConfig(h, &RequestHeaderConfig{
		UsernameHeaders:     []string{"X-Original-User"},
		ExtraHeaderPrefixes: []string{"X-Original-Extra-"},
	})
	want := http.Header{"X-Unrelated": {"kept"}}
	if diff := cmp.Diff(want, h); len(diff) > 0 {
		t.Errorf("unexpected final headers (-want +got):\n%s", diff)
	}
}

// newTestClientCertsFromSameCA returns a root pool and one client certificate per given
// common name, all signed by the same CA.
func newTestClientCertsFromSameCA(t *testing.T, commonNames ...string) (*x509.CertPool, []*x509.Certificate) {
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

	var clientCerts []*x509.Certificate
	for i, commonName := range commonNames {
		clientKey, err := rsa.GenerateKey(rand.Reader, 2048)
		if err != nil {
			t.Fatal(err)
		}
		clientTemplate := &x509.Certificate{
			SerialNumber: big.NewInt(int64(i + 2)),
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
		clientCerts = append(clientCerts, clientCert)
	}

	roots := x509.NewCertPool()
	roots.AddCert(caCert)
	return roots, clientCerts
}
