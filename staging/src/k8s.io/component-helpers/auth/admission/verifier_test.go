/*
Copyright The Kubernetes Authors.

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

package admission

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/textlogger"
)

func TestAllowsAPIGroup(t *testing.T) {
	for _, tc := range []struct {
		name   string
		groups []string
		group  string
		want   bool
	}{
		{name: "exact", groups: []string{"apps"}, group: "apps", want: true},
		{name: "wildcard", groups: []string{"*"}, group: "example.com", want: true},
		{name: "wildcard core", groups: []string{"*"}, group: "", want: true},
		{name: "empty", groups: nil, group: "apps"},
		{name: "other", groups: []string{"extensions"}, group: "apps"},
		{name: "multiple unrelated", groups: []string{"extensions", "batch"}, group: "apps"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := allowsAPIGroup(tc.groups, tc.group); got != tc.want {
				t.Fatalf("allowsAPIGroup(%q, %q) = %t, want %t", tc.groups, tc.group, got, tc.want)
			}
		})
	}
}

// TestVerifierVerify exercises the verifier path without allowing an
// unauthorized group or nil request through the admission gate.
func TestVerifierVerify(t *testing.T) {
	ts := newOIDCTestServer(t)
	vv := ts.verifier(t)

	t.Run("valid token and matching group -> nil", func(t *testing.T) {
		token := ts.sign(t, ts.baseClaims())
		if err := vv.Verify(context.Background(), admissionReview(testGroup).Request, token); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("group not authorized -> generic failure", func(t *testing.T) {
		token := ts.sign(t, ts.baseClaims())
		err := vv.Verify(context.Background(), admissionReview("batch").Request, token)
		if err == nil || err.Error() != "webhook token verification failed" {
			t.Fatalf("want generic verification failure, got %v", err)
		}
	})

	t.Run("nil request -> fails closed", func(t *testing.T) {
		token := ts.sign(t, ts.baseClaims())
		err := vv.Verify(context.Background(), nil, token)
		if err == nil || err.Error() != "webhook token verification failed" {
			t.Fatalf("want generic verification failure for a nil request, got %v", err)
		}
	})
}

func TestLibraryKeyRotationAndAvailability(t *testing.T) {
	first := newOIDCTestServer(t)
	second := newECDSAOIDCIssuer(t)
	var keys atomic.Value
	keys.Store(testJWKS(t, first.server, "/keys"))
	var status atomic.Int32
	status.Store(http.StatusOK)
	var keyRequests atomic.Int32
	var server *httptest.Server
	server = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/.well-known/openid-configuration" {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"issuer":                                server.URL,
				"jwks_uri":                              server.URL + "/keys",
				"id_token_signing_alg_values_supported": []string{"RS256", "ES256"},
			})
			return
		}
		if r.URL.Path != "/keys" {
			http.NotFound(w, r)
			return
		}
		keyRequests.Add(1)
		w.WriteHeader(int(status.Load()))
		_, _ = w.Write(keys.Load().([]byte))
	}))
	defer server.Close()
	newVerifier := func() (*VerifierForTest, error) {
		return NewVerifierForTest(context.Background(), WithIssuer(server.URL, testAudience, server.Client()))
	}
	v, err := newVerifier()
	if err != nil {
		t.Fatal(err)
	}
	if got := keyRequests.Load(); got != 0 {
		t.Fatalf("construction fetched signing keys %d times, want 0", got)
	}
	firstClaims := first.baseClaims()
	firstClaims["iss"] = server.URL
	if err := v.Verify(context.Background(), admissionReview(testGroup).Request, first.sign(t, firstClaims)); err != nil {
		t.Fatalf("first signing key: %v", err)
	}
	if got := keyRequests.Load(); got != 1 {
		t.Fatalf("first verification requests = %d, want 1 (lazy key fetch)", got)
	}
	keys.Store(testJWKS(t, second.server, "/openid/v1/jwks"))
	secondClaims := second.validWebhookClaims(testAudience, "validatingwebhookconfiguration", "vwc", "uid", []string{testGroup})
	secondClaims["iss"] = server.URL
	token := second.sign(t, secondClaims)
	if err := v.Verify(context.Background(), admissionReview(testGroup).Request, token); err != nil {
		t.Fatalf("rotated signing key: %v", err)
	}
	if got := keyRequests.Load(); got != 2 {
		t.Fatalf("rotation requests = %d, want 2", got)
	}
	status.Store(http.StatusServiceUnavailable)
	before := keyRequests.Load()
	if err := v.Verify(context.Background(), admissionReview(testGroup).Request, token); err != nil {
		t.Fatalf("cached signing keys unusable during endpoint outage: %v", err)
	}
	if keyRequests.Load() != before {
		t.Fatal("cached verification fetched keys during endpoint outage")
	}
	status.Store(http.StatusOK)
	keys.Store([]byte("not JSON"))
	fresh, err := newVerifier()
	if err != nil {
		t.Fatalf("lazy verifier construction: %v", err)
	}
	if err := fresh.Verify(context.Background(), admissionReview(testGroup).Request, token); err == nil {
		t.Fatal("real verification accepted malformed signing keys")
	}
	keys.Store(testJWKS(t, second.server, "/openid/v1/jwks"))
	status.Store(http.StatusServiceUnavailable)
	before = keyRequests.Load()
	unavailable, err := newVerifier()
	if err != nil || unavailable == nil {
		t.Fatalf("discovery with unavailable signing keys = (%v, %v), want a lazy verifier", unavailable, err)
	}
	if keyRequests.Load() != before {
		t.Fatal("construction fetched unavailable signing keys")
	}
	if err := unavailable.Verify(context.Background(), admissionReview(testGroup).Request, token); err == nil {
		t.Fatal("verification accepted unavailable signing keys")
	}
	if keyRequests.Load() <= before {
		t.Fatal("verification did not fetch unavailable signing keys")
	}
}

func TestVerifierErrorsAreGeneric(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	good := ts.sign(t, ts.baseClaims())
	invalidClaims := ts.baseClaims()
	delete(invalidClaims, "kubernetes.io")
	generic := v.Verify(context.Background(), nil, good)
	for _, tc := range []struct {
		name  string
		token string
		req   *admissionv1.AdmissionRequest
	}{
		{name: "signature", token: "invalid", req: admissionReview(testGroup).Request},
		{name: "Kubernetes claims", token: ts.sign(t, invalidClaims), req: admissionReview(testGroup).Request},
		{name: "group", token: good, req: admissionReview("batch").Request},
		{name: "nil request", token: good},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := v.Verify(context.Background(), tc.req, tc.token)
			//nolint:errorlint // Wrapped errors must fail this identity assertion.
			if err == nil || err != generic || err.Error() != "webhook token verification failed" {
				t.Fatalf("verification error = %v, want the same generic failure", err)
			}
		})
	}
	if err := v.Verify(context.Background(), admissionReview(testGroup).Request, good); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	//nolint:errorlint // Wrapped errors must fail this identity assertion.
	if err := v.Verify(ctx, admissionReview(testGroup).Request, good); err != generic {
		t.Fatalf("cancelled verification = %v, want generic failure", err)
	}
}

// TestVerifierVerify_ZeroDecode asserts the decoded entry point cannot read
// an HTTP body because Verifier.Verify takes only an AdmissionRequest and token.
func TestVerifierVerify_ZeroDecode(t *testing.T) {
	ts := newOIDCTestServer(t)
	vv := ts.verifier(t)
	token := ts.sign(t, ts.baseClaims())

	req := admissionReview(testGroup).Request
	if err := vv.Verify(context.Background(), req, token); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// A nil request fails closed with the generic error, still without any decode.
	err := vv.Verify(context.Background(), nil, token)
	if err == nil || err.Error() != "webhook token verification failed" {
		t.Fatalf("nil request: want generic verification failure, got %v", err)
	}
}

// TestNewVerifier_InClusterExplicitAudience verifies that the decoded in-cluster
// path enforces the explicitly configured audience.
func TestNewVerifier_InClusterExplicitAudience(t *testing.T) {
	ts := newOIDCTestServer(t)

	v, err := ts.localVerifier(testAudience)
	if err != nil {
		t.Fatalf("NewVerifier: %v", err)
	}
	token := ts.sign(t, ts.baseClaims())
	if err := v.Verify(context.Background(), admissionReview(testGroup).Request, token); err != nil {
		t.Errorf("unexpected verification error: %v", err)
	}

	// A token carrying the wrong audience is denied fail-closed (the bound audience
	// is enforced natively as the go-oidc ClientID).
	wrong := ts.baseClaims()
	wrong["aud"] = []string{"someone.else.example.com"}
	err = v.Verify(context.Background(), admissionReview(testGroup).Request, ts.sign(t, wrong))
	if err == nil || err.Error() != "webhook token verification failed" {
		t.Errorf("wrong audience: want generic verification failure, got %v", err)
	}
}

func TestVerifier_SubresourceKindAndObjectGroups(t *testing.T) {
	ts := newOIDCTestServer(t)
	verifier, err := NewVerifierForTest(context.Background(), WithIssuer(ts.issuer, testAudience, ts.client()))
	if err != nil {
		t.Fatalf("construct token verifier: %v", err)
	}
	token := ts.sign(t, ts.baseClaims())
	appsResource := metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}
	appsKind := metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}
	scaleKind := metav1.GroupVersionKind{Group: "autoscaling", Version: "v1", Kind: "Scale"}
	for _, tc := range []struct {
		name     string
		kind     metav1.GroupVersionKind
		sub      string
		object   string
		old      string
		wantDeny bool
	}{
		{
			name: "legitimate deployments scale subresource",
			kind: scaleKind, sub: "scale", object: `{"apiVersion":"autoscaling/v1","kind":"Scale"}`,
		},
		{
			name: "scale subresource rejects unsupported kind version",
			kind: metav1.GroupVersionKind{Group: "autoscaling", Version: "v2", Kind: "Scale"},
			sub:  "scale", object: `{"apiVersion":"autoscaling/v2","kind":"Scale"}`, wantDeny: true,
		},
		{
			name: "ordinary resource cannot switch kind groups",
			kind: scaleKind, object: `{"apiVersion":"autoscaling/v1","kind":"Scale"}`, wantDeny: true,
		},
		{
			name: "forged scale subresource cannot carry a Role",
			kind: metav1.GroupVersionKind{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "Role"},
			sub:  "scale", object: `{"apiVersion":"rbac.authorization.k8s.io/v1","kind":"Role"}`, wantDeny: true,
		},
		{
			name: "unknown subresource cannot switch to autoscaling group",
			kind: scaleKind, sub: "other", object: `{"apiVersion":"autoscaling/v1","kind":"Scale"}`, wantDeny: true,
		},
		{
			name: "subresource object differs from admitted Scale kind",
			kind: scaleKind, sub: "scale", object: `{"apiVersion":"apps/v1","kind":"Deployment"}`, wantDeny: true,
		},
		{
			name: "subresource without a kind cannot claim another object group",
			sub:  "scale", object: `{"apiVersion":"autoscaling/v1","kind":"Scale"}`, wantDeny: true,
		},
		{
			name: "old object mismatches scale kind",
			kind: scaleKind, sub: "scale", object: `{"apiVersion":"autoscaling/v1","kind":"Scale"}`,
			old: `{"apiVersion":"apps/v1","kind":"Deployment"}`, wantDeny: true,
		},
		{
			name: "ordinary object Kind disagrees with AdmissionRequest Kind",
			kind: appsKind, object: `{"apiVersion":"apps/v1","kind":"Pod"}`, wantDeny: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			operation := admissionv1.Create
			if tc.sub == "scale" || tc.old != "" {
				operation = admissionv1.Update
			}
			request := admissionRequest(tc.kind, appsResource, tc.sub, operation)
			request.Object.Raw = []byte(tc.object)
			if operation == admissionv1.Update {
				request.OldObject.Raw = bytes.Clone(request.Object.Raw)
			}
			if tc.old != "" {
				request.OldObject.Raw = []byte(tc.old)
			}
			err := verifier.Verify(context.Background(), request, token)
			if tc.wantDeny {
				if err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("verification = %v, want generic denial", err)
				}
			} else if err != nil {
				t.Fatalf("valid subresource request denied: %v", err)
			}
		})
	}
}

func TestVerifier_DecodedObjectsMatchAdmittedKind(t *testing.T) {
	ts := newOIDCTestServer(t)
	verifier, err := NewVerifierForTest(context.Background(), WithIssuer(ts.issuer, testAudience, ts.client()))
	if err != nil {
		t.Fatalf("construct token verifier: %v", err)
	}
	token := ts.sign(t, ts.baseClaims())
	object := func(apiVersion, kind string) runtime.Object {
		return &unstructured.Unstructured{Object: map[string]any{"apiVersion": apiVersion, "kind": kind}}
	}
	for _, tc := range []struct {
		name     string
		current  runtime.Object
		old      runtime.Object
		wantDeny bool
	}{
		{name: "matching current and old objects", current: object("apps/v1", "Deployment"), old: object("apps/v1", "Deployment")},
		{name: "wrong current group", current: object("batch/v1", "Job"), wantDeny: true},
		{name: "wrong old group", current: object("apps/v1", "Deployment"), old: object("batch/v1", "Job"), wantDeny: true},
		{name: "missing object type metadata", current: &unstructured.Unstructured{Object: map[string]any{"metadata": map[string]any{"name": "missing"}}}, wantDeny: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			operation := admissionv1.Create
			if tc.old != nil {
				operation = admissionv1.Update
			}
			req := admissionRequest(
				metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
				metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
				"", operation,
			)
			req.Object = runtime.RawExtension{Object: tc.current}
			if tc.old != nil {
				req.OldObject = runtime.RawExtension{Object: tc.old}
			}
			err := verifier.Verify(context.Background(), req, token)
			if tc.wantDeny {
				if err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("verification = %v, want generic denial", err)
				}
			} else if err != nil {
				t.Fatalf("matching decoded objects denied: %v", err)
			}
		})
	}
}

func TestVerifier_CoreCrossGroupSubresources(t *testing.T) {
	ts := newOIDCTestServer(t)
	verifier, err := NewVerifierForTest(context.Background(), WithIssuer(ts.issuer, testAudience, ts.client()))
	if err != nil {
		t.Fatalf("construct token verifier: %v", err)
	}
	claims := ts.baseClaims()
	claims["kubernetes.io"].(map[string]any)["attestations"] = map[string]any{
		admissionReviewAPIGroupsClaimKey: []string{"*"},
	}
	token := ts.sign(t, claims)
	for _, tc := range []struct {
		name                                   string
		resource, resourceVersion, subresource string
		group, kindVersion, kind               string
		object                                 string
		wantDeny                               bool
	}{
		{
			name: "Pod eviction", resource: "pods", resourceVersion: "v1", subresource: "eviction", group: "policy", kindVersion: "v1", kind: "Eviction",
			object: `{"apiVersion":"policy/v1","kind":"Eviction"}`,
		},
		{
			name: "ServiceAccount token", resource: "serviceaccounts", resourceVersion: "v1", subresource: "token", group: "authentication.k8s.io", kindVersion: "v1", kind: "TokenRequest",
			object: `{"apiVersion":"authentication.k8s.io/v1","kind":"TokenRequest"}`,
		},
		{
			name: "Pod eviction rejects unsupported resource version", resource: "pods", resourceVersion: "v2", subresource: "eviction", group: "policy", kindVersion: "v1", kind: "Eviction",
			object: `{"apiVersion":"policy/v1","kind":"Eviction"}`, wantDeny: true,
		},
		{
			name: "ServiceAccount token rejects unsupported resource version", resource: "serviceaccounts", resourceVersion: "v2", subresource: "token", group: "authentication.k8s.io", kindVersion: "v1", kind: "TokenRequest",
			object: `{"apiVersion":"authentication.k8s.io/v1","kind":"TokenRequest"}`, wantDeny: true,
		},
		{
			name: "Pod eviction rejects unsupported kind version", resource: "pods", resourceVersion: "v1", subresource: "eviction", group: "policy", kindVersion: "v2", kind: "Eviction",
			object: `{"apiVersion":"policy/v2","kind":"Eviction"}`, wantDeny: true,
		},
		{
			name: "ServiceAccount token rejects unsupported kind version", resource: "serviceaccounts", resourceVersion: "v1", subresource: "token", group: "authentication.k8s.io", kindVersion: "v2", kind: "TokenRequest",
			object: `{"apiVersion":"authentication.k8s.io/v2","kind":"TokenRequest"}`, wantDeny: true,
		},
		{
			name: "Pod eviction cannot claim a TokenRequest Kind", resource: "pods", resourceVersion: "v1", subresource: "eviction", group: "authentication.k8s.io", kindVersion: "v1", kind: "TokenRequest",
			object: `{"apiVersion":"authentication.k8s.io/v1","kind":"TokenRequest"}`, wantDeny: true,
		},
		{
			name: "Secret cannot spoof a Pod eviction", resource: "secrets", resourceVersion: "v1", subresource: "eviction", group: "policy", kindVersion: "v1", kind: "Eviction",
			object: `{"apiVersion":"policy/v1","kind":"Eviction"}`, wantDeny: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := admissionRequest(
				metav1.GroupVersionKind{Group: tc.group, Version: tc.kindVersion, Kind: tc.kind},
				metav1.GroupVersionResource{Version: tc.resourceVersion, Resource: tc.resource},
				tc.subresource, admissionv1.Create,
			)
			request.Object.Raw = []byte(tc.object)
			err := verifier.Verify(context.Background(), request, token)
			if tc.wantDeny {
				if err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("cross-group spoof: verification = %v, want generic denial", err)
				}
			} else if err != nil {
				t.Fatalf("legitimate cross-group subresource denied: %v", err)
			}
		})
	}
}

func TestVerifier_RejectsMissingObjectTypeMetaForCoreGroup(t *testing.T) {
	ts := newOIDCTestServer(t)
	verifier, err := NewVerifierForTest(context.Background(), WithIssuer(ts.issuer, testAudience, ts.client()))
	if err != nil {
		t.Fatalf("construct token verifier: %v", err)
	}
	claims := ts.baseClaims()
	claims["kubernetes.io"].(map[string]any)["attestations"] = map[string]any{
		admissionReviewAPIGroupsClaimKey: []string{"*"},
	}
	token := ts.sign(t, claims)
	for _, tc := range []struct{ name, object string }{
		{name: "JSON null", object: `null`},
		{name: "empty object", object: `{}`},
		{name: "versionless apiVersion", object: `{"apiVersion":"/","kind":"Pod"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := admissionRequest(
				metav1.GroupVersionKind{Version: "v1", Kind: "Pod"},
				metav1.GroupVersionResource{Version: "v1", Resource: "pods"},
				"", admissionv1.Create,
			)
			request.Object.Raw = []byte(tc.object)
			if err := verifier.Verify(context.Background(), request, token); err == nil || err.Error() != "webhook token verification failed" {
				t.Fatalf("missing object type metadata: got %v, want generic denial", err)
			}
		})
	}
}

func TestVerifierOperationShapes(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	claims := ts.baseClaims()
	claims["kubernetes.io"].(map[string]any)["attestations"] = map[string]any{admissionReviewAPIGroupsClaimKey: []string{"*"}}
	token := ts.sign(t, claims)
	appsKind := metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}
	appsResource := metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}
	for _, tc := range []struct {
		name      string
		operation admissionv1.Operation
		mutate    func(*admissionv1.AdmissionRequest)
	}{
		{name: "create", operation: admissionv1.Create},
		{name: "update", operation: admissionv1.Update},
		{name: "delete old only", operation: admissionv1.Delete},
		{name: "connect with options", operation: admissionv1.Connect, mutate: func(r *admissionv1.AdmissionRequest) {
			*r = *admissionRequest(metav1.GroupVersionKind{Version: "v1", Kind: "PodExecOptions"}, metav1.GroupVersionResource{Version: "v1", Resource: "pods"}, "exec", admissionv1.Connect)
		}},
		{name: "connect without options", operation: admissionv1.Connect, mutate: func(r *admissionv1.AdmissionRequest) { r.Object = runtime.RawExtension{} }},
		{name: "core group", operation: admissionv1.Create, mutate: func(r *admissionv1.AdmissionRequest) {
			*r = *admissionRequest(metav1.GroupVersionKind{Version: "v1", Kind: "Pod"}, metav1.GroupVersionResource{Version: "v1", Resource: "pods"}, "", admissionv1.Create)
		}},
		{name: "CRD scale version differs from parent", operation: admissionv1.Update, mutate: func(r *admissionv1.AdmissionRequest) {
			*r = *admissionRequest(metav1.GroupVersionKind{Group: "autoscaling", Version: "v1", Kind: "Scale"}, metav1.GroupVersionResource{Group: "example.com", Version: "v2", Resource: "widgets"}, "scale", admissionv1.Update)
		}},
		{name: "custom future operation", operation: "CUSTOM"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := admissionRequest(appsKind, appsResource, "", tc.operation)
			if tc.mutate != nil {
				tc.mutate(req)
			}
			if err := v.Verify(context.Background(), req, token); err != nil {
				t.Fatalf("legitimate operation denied: %v", err)
			}
			raw, err := json.Marshal(&admissionv1.AdmissionReview{TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"}, Request: req})
			if err != nil {
				t.Fatal(err)
			}
			spy := newSpyHandler()
			h := NewHandlerForTest(v, spy.serve)
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, newRequest(t, raw, token))
			var result admissionv1.AdmissionReview
			if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if rec.Code != http.StatusOK || result.Response == nil || !result.Response.Allowed || result.Response.UID != req.UID || !spy.wasReached() {
				t.Fatalf("HTTP verification: status=%d response=%+v callback=%t", rec.Code, result.Response, spy.wasReached())
			}
		})
	}
}

type metadataOnlyObject struct {
	metav1.TypeMeta
	NonJSON chan struct{} `json:"nonJSON"`
}

func (o *metadataOnlyObject) DeepCopyObject() runtime.Object {
	if o == nil {
		return nil
	}
	copy := *o
	return &copy
}

func TestVerifierMetadataRepresentations(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	token := ts.sign(t, ts.baseClaims())
	decoded := func(apiVersion, kind string) runtime.Object {
		return &unstructured.Unstructured{Object: map[string]any{"apiVersion": apiVersion, "kind": kind}}
	}
	for _, tc := range []struct {
		name   string
		mutate func(*admissionv1.AdmissionRequest)
		deny   bool
	}{
		{name: "complete metadata"},
		{name: "missing kind", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Kind = metav1.GroupVersionKind{} }},
		{name: "missing kind version", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Kind.Version = "" }},
		{name: "missing kind name", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Kind.Kind = "" }},
		{name: "missing resource version", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Resource.Version = "" }},
		{name: "missing resource name", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Resource.Resource = "" }},
		{name: "null object", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Object.Raw = []byte(" \nnull\t ") }},
		{name: "malformed raw object", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Object.Raw = []byte("{") }},
		{name: "missing object version", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Object.Raw = []byte(`{"kind":"Deployment"}`) }},
		{name: "missing object kind", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Object.Raw = []byte(`{"apiVersion":"apps/v1"}`) }},
		{name: "wrong object version", deny: true, mutate: func(r *admissionv1.AdmissionRequest) {
			r.Object.Raw = []byte(`{"apiVersion":"apps/v2","kind":"Deployment"}`)
		}},
		{name: "wrong object kind", deny: true, mutate: func(r *admissionv1.AdmissionRequest) {
			r.Object.Raw = []byte(`{"apiVersion":"apps/v1","kind":"StatefulSet"}`)
		}},
		{name: "matching raw and decoded", mutate: func(r *admissionv1.AdmissionRequest) { r.Object.Object = decoded("apps/v1", "Deployment") }},
		{name: "conflicting raw and decoded", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Object.Object = decoded("batch/v1", "Job") }},
		{name: "null raw and present decoded", deny: true, mutate: func(r *admissionv1.AdmissionRequest) {
			r.Object.Raw = []byte("null")
			r.Object.Object = decoded("apps/v1", "Deployment")
		}},
		{name: "decoded metadata only", mutate: func(r *admissionv1.AdmissionRequest) {
			r.Object = runtime.RawExtension{Object: &metadataOnlyObject{TypeMeta: metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"}, NonJSON: make(chan struct{})}}
		}},
		{name: "original equivalent version differs", mutate: func(r *admissionv1.AdmissionRequest) {
			r.RequestKind = &metav1.GroupVersionKind{Group: "extensions", Version: "v1beta1", Kind: "Deployment"}
			r.RequestResource = &metav1.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "deployments"}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := admissionReview(testGroup).Request
			if tc.mutate != nil {
				tc.mutate(req)
			}
			err := v.Verify(context.Background(), req, token)
			if tc.deny {
				if err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("Verify = %v, want generic denial", err)
				}
			} else if err != nil {
				t.Fatalf("matching metadata denied: %v", err)
			}
		})
	}
}

func TestAdmissionRequestKindDiagnostics(t *testing.T) {
	var output bytes.Buffer
	logger := textlogger.NewLogger(textlogger.NewConfig(textlogger.Verbosity(2), textlogger.Output(&output)))
	ctx := klog.NewContext(context.Background(), logger)
	req := &admissionv1.AdmissionRequest{
		UID:       "diagnostic-request-uid",
		Kind:      metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
		Resource:  metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
		Operation: admissionv1.Create,
		Name:      "private-resource-name",
		Namespace: "private-namespace",
		UserInfo: authenticationv1.UserInfo{
			Username: "private-user-name",
			UID:      "private-user-uid",
		},
		Object:  runtime.RawExtension{Raw: []byte(`{"apiVersion":"batch/v1","kind":"Job","spec":{"credential":"private-payload-credential"}}`)},
		Options: runtime.RawExtension{Raw: []byte(`{"credential":"private-option-credential"}`)},
	}
	err := (&verifier{}).checkDecodedRequest(ctx, req, []string{"apps"})
	if err == nil || err.Error() != "webhook token verification failed" {
		t.Fatalf("caller error = %v, want generic verification failure", err)
	}
	log := output.String()
	for _, want := range []string{
		`"Admission request rejected"`,
		`reason="invalid admission request"`,
		`requestUID="diagnostic-request-uid"`,
		`operation="CREATE"`,
		`resource="apps/v1, Resource=deployments"`,
		`detail="object GVK batch/v1, Kind=Job differs from the admitted kind"`,
	} {
		if !strings.Contains(log, want) {
			t.Errorf("log missing %q: %s", want, log)
		}
	}
	for _, private := range []string{
		req.Name, req.Namespace, req.UserInfo.Username, req.UserInfo.UID,
		"private-option-credential", "private-payload-credential",
	} {
		if strings.Contains(log, private) {
			t.Errorf("log leaked %q: %s", private, log)
		}
	}
}

type staticSubresourceMismatchProvider struct {
	expectedMismatches ExpectedSubresourceGroupMismatchMap
	err                error
}

func (p *staticSubresourceMismatchProvider) Build(context.Context) (ExpectedSubresourceGroupMismatchMap, error) {
	return p.expectedMismatches, p.err
}

func TestAllowedSubresourceGroupMismatchList_MutationIsolated(t *testing.T) {
	first := AllowedSubresourceGroupMismatchList()
	if len(first) != 3 {
		t.Fatalf("len(AllowedSubresourceGroupMismatchList()) = %d, want 3", len(first))
	}
	scaleKey := SubresourceGroupMismatchKey{Subresource: "scale"}
	first[scaleKey][schema.GroupVersionKind{Group: "evil.example.com", Version: "v1", Kind: "Scale"}] = struct{}{}
	delete(first, SubresourceGroupMismatchKey{Resource: coreV1PodsGVR, Subresource: "eviction"})

	second := AllowedSubresourceGroupMismatchList()
	if _, ok := second[scaleKey][schema.GroupVersionKind{Group: "evil.example.com", Version: "v1", Kind: "Scale"}]; ok {
		t.Fatal("mutating inner GVK set in AllowedSubresourceGroupMismatchList affected subsequent calls")
	}
	if _, ok := second[SubresourceGroupMismatchKey{Resource: coreV1PodsGVR, Subresource: "eviction"}]; !ok {
		t.Fatal("deleting key in AllowedSubresourceGroupMismatchList affected subsequent calls")
	}
}

func TestWithSubresourceGroupMismatches(t *testing.T) {
	ts := newOIDCTestServer(t)
	claims := ts.baseClaims()
	claims["kubernetes.io"].(map[string]any)["attestations"] = map[string]any{
		admissionReviewAPIGroupsClaimKey: []string{"pets.example.com", "apps"},
	}
	token := ts.sign(t, claims)

	customResource := metav1.GroupVersionResource{Group: "pets.example.com", Version: "v1", Resource: "kittens"}
	customKind := metav1.GroupVersionKind{Group: "adoption.example.com", Version: "v1", Kind: "AdoptionRequest"}
	customReq := admissionRequest(customKind, customResource, "adopt", admissionv1.Create)
	customReq.Object.Raw = []byte(`{"apiVersion":"adoption.example.com/v1","kind":"AdoptionRequest"}`)

	defaultVerifier, err := NewVerifierForTest(context.Background(), WithIssuer(ts.issuer, testAudience, ts.client()))
	if err != nil {
		t.Fatalf("construct default verifier: %v", err)
	}
	if err := defaultVerifier.Verify(context.Background(), customReq, token); err == nil {
		t.Fatal("default verifier accepted custom cross-group subresource without override")
	}

	extended := AllowedSubresourceGroupMismatchList()
	extended[SubresourceGroupMismatchKey{Resource: customResource, Subresource: "adopt"}] = map[schema.GroupVersionKind]struct{}{
		schema.GroupVersionKind(customKind): {},
	}
	customVerifier, err := NewVerifierForTest(
		context.Background(),
		WithIssuer(ts.issuer, testAudience, ts.client()),
		WithSubresourceGroupMismatches(&staticSubresourceMismatchProvider{expectedMismatches: extended}),
	)
	if err != nil {
		t.Fatalf("construct custom verifier: %v", err)
	}
	if err := customVerifier.Verify(context.Background(), customReq, token); err != nil {
		t.Fatalf("custom verifier rejected configured cross-group subresource: %v", err)
	}
	scaleReq := admissionRequest(
		metav1.GroupVersionKind{Group: "autoscaling", Version: "v1", Kind: "Scale"},
		metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
		"scale", admissionv1.Update,
	)
	if err := customVerifier.Verify(context.Background(), scaleReq, token); err != nil {
		t.Fatalf("seeded default scale subresource rejected: %v", err)
	}

	if _, err := NewVerifierForTest(context.Background(), WithIssuer(ts.issuer, testAudience, ts.client()), WithSubresourceGroupMismatches(nil)); err == nil {
		t.Fatal("nil WithSubresourceGroupMismatches provider succeeded, want error")
	}
	if _, err := NewVerifierForTest(
		context.Background(),
		WithIssuer(ts.issuer, testAudience, ts.client()),
		WithSubresourceGroupMismatches(&staticSubresourceMismatchProvider{err: errors.New("discovery unavailable")}),
	); err == nil {
		t.Fatal("failing WithSubresourceGroupMismatches provider succeeded, want startup error")
	}
}

func TestDiscoverSubresourceGroupMismatches(t *testing.T) {
	var discoveryRequests atomic.Int32
	var failDiscovery atomic.Bool
	aggGroups := apidiscoveryv2.APIGroupDiscoveryList{
		TypeMeta: metav1.TypeMeta{APIVersion: "apidiscovery.k8s.io/v2", Kind: "APIGroupDiscoveryList"},
		Items: []apidiscoveryv2.APIGroupDiscovery{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "pets.example.com"},
				Versions: []apidiscoveryv2.APIVersionDiscovery{
					{
						Version:   "v1",
						Freshness: apidiscoveryv2.DiscoveryFreshnessCurrent,
						Resources: []apidiscoveryv2.APIResourceDiscovery{
							{
								Resource:     "kittens",
								ResponseKind: &metav1.GroupVersionKind{Group: "pets.example.com", Version: "v1", Kind: "Kitten"},
								Scope:        apidiscoveryv2.ScopeNamespace,
								Verbs:        []string{"create", "get"},
								Subresources: []apidiscoveryv2.APISubresourceDiscovery{
									{
										Subresource:  "status",
										ResponseKind: &metav1.GroupVersionKind{Group: "pets.example.com", Version: "v1", Kind: "Kitten"},
										Verbs:        []string{"update"},
									},
									{
										Subresource:  "adopt",
										ResponseKind: &metav1.GroupVersionKind{Group: "adoption.example.com", Version: "v1", Kind: "AdoptionRequest"},
										Verbs:        []string{"create"},
									},
									{
										Subresource:  "scale",
										ResponseKind: &metav1.GroupVersionKind{Group: "autoscaling", Version: "v1", Kind: "Scale"},
										Verbs:        []string{"get", "update"},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	coreGroups := apidiscoveryv2.APIGroupDiscoveryList{
		TypeMeta: metav1.TypeMeta{APIVersion: "apidiscovery.k8s.io/v2", Kind: "APIGroupDiscoveryList"},
	}

	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		discoveryRequests.Add(1)
		if failDiscovery.Load() {
			http.Error(w, "discovery down", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList")
		switch r.URL.Path {
		case "/api":
			_ = json.NewEncoder(w).Encode(coreGroups)
		case "/apis":
			_ = json.NewEncoder(w).Encode(aggGroups)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	cfg := &rest.Config{
		Host:      srv.URL,
		Transport: srv.Client().Transport,
	}
	discClient, err := discovery.NewDiscoveryClientForConfig(cfg)
	if err != nil {
		t.Fatalf("NewDiscoveryClientForConfig: %v", err)
	}
	cached := memory.NewMemCacheClient(discClient)

	mismatches, err := DiscoverSubresourceGroupMismatches(context.Background(), cached)
	if err != nil {
		t.Fatalf("DiscoverSubresourceGroupMismatches: %v", err)
	}
	firstCallRequests := discoveryRequests.Load()
	if firstCallRequests == 0 {
		t.Fatal("expected discovery requests on initial populate")
	}

	adoptKey := SubresourceGroupMismatchKey{
		Resource:    metav1.GroupVersionResource{Group: "pets.example.com", Version: "v1", Resource: "kittens"},
		Subresource: "adopt",
	}
	wantAdoptKind := schema.GroupVersionKind{Group: "adoption.example.com", Version: "v1", Kind: "AdoptionRequest"}
	if _, ok := mismatches[adoptKey][wantAdoptKind]; !ok {
		t.Fatalf("discovered mismatches missing %+v -> %+v: %+v", adoptKey, wantAdoptKind, mismatches)
	}

	// Subsequent call without Invalidate uses the cached discovery response.
	if _, err := DiscoverSubresourceGroupMismatches(context.Background(), cached); err != nil {
		t.Fatalf("cached DiscoverSubresourceGroupMismatches: %v", err)
	}
	if discoveryRequests.Load() != firstCallRequests {
		t.Fatalf("cached discovery made extra requests: got %d, want %d", discoveryRequests.Load(), firstCallRequests)
	}

	// Failing discovery after cache invalidation fails closed.
	failDiscovery.Store(true)
	cached.Invalidate()
	if _, err := DiscoverSubresourceGroupMismatches(context.Background(), cached); err == nil {
		t.Fatal("expected error when discovery fails, got nil")
	}
	if _, err := DiscoverSubresourceGroupMismatches(context.Background(), nil); err == nil {
		t.Fatal("expected error when cached client is nil, got nil")
	}
}
