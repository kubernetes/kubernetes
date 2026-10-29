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
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/textlogger"
)

// TestBearerToken_Table exercises the header parser directly across the scheme
// and token shapes it must accept or reject. The scheme match is
// case-insensitive (RFC 7235); an empty or whitespace-only token is rejected so
// a bare "Bearer" header cannot slip through as a present-but-empty credential.
func TestBearerToken_Table(t *testing.T) {
	tests := []struct {
		name      string
		header    string // "" means no Authorization header at all
		wantToken string
		wantOK    bool
	}{
		{name: "valid bearer", header: "Bearer abc.def.ghi", wantToken: "abc.def.ghi", wantOK: true},
		{name: "lowercase scheme is accepted", header: "bearer abc", wantToken: "abc", wantOK: true},
		{name: "mixed-case scheme is accepted", header: "BeArEr abc", wantToken: "abc", wantOK: true},
		{name: "trailing header whitespace is trimmed", header: "Bearer abc   ", wantToken: "abc", wantOK: true},
		{name: "extra space before the token is rejected (matches apiserver)", header: "Bearer    abc", wantOK: false},
		{name: "no header", header: "", wantOK: false},
		{name: "non-bearer scheme", header: "Basic dXNlcjpwYXNz", wantOK: false},
		{name: "bearer with empty token", header: "Bearer ", wantOK: false},
		{name: "bearer with whitespace-only token", header: "Bearer      ", wantOK: false},
		{name: "shorter than the scheme prefix", header: "Bear", wantOK: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, "/validate", nil)
			if tc.header != "" {
				r.Header.Set("Authorization", tc.header)
			}
			token, ok := bearerToken(r)
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if ok && token != tc.wantToken {
				t.Errorf("token = %q, want %q", token, tc.wantToken)
			}
		})
	}
}

func TestWithTokenVerificationConstruction(t *testing.T) {
	var requests atomic.Int32
	var server *httptest.Server
	server = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/.well-known/openid-configuration" {
			_ = json.NewEncoder(w).Encode(map[string]any{"issuer": server.URL, "jwks_uri": server.URL + localJWKSPath})
			return
		}
		if r.URL.Path != localJWKSPath {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"keys":[]}`))
	}))
	defer server.Close()
	var optionsApplied int
	option := func(b *builder) {
		optionsApplied++
		b.remoteSet = true
		b.issuer = server.URL
		b.audience = "https://webhook.example/validate"
		b.httpClient = server.Client()
	}
	if h, err := WithTokenVerification(context.Background(), nil, option); h != nil || err == nil {
		t.Fatalf("nil callback = (%v, %v), want (nil, error)", h, err)
	}
	if optionsApplied != 0 || requests.Load() != 0 {
		t.Fatal("nil callback applied options or performed I/O")
	}
	h, err := WithTokenVerification(context.Background(), func(context.Context, *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		return &admissionv1.AdmissionResponse{Allowed: true}
	}, option)
	if err != nil || h == nil {
		t.Fatalf("construction = (%v, %v)", h, err)
	}
	if optionsApplied != 1 || requests.Load() != 1 {
		t.Fatalf("options applied %d times, discovery requests %d; want 1 and 1", optionsApplied, requests.Load())
	}
}

func TestAdmissionRequestPresenceDiagnostics(t *testing.T) {
	for _, tc := range []struct {
		name       string
		operation  admissionv1.Operation
		object     bool
		oldObject  bool
		wantDetail string
	}{
		{name: "create valid", operation: admissionv1.Create, object: true},
		{name: "create missing object", operation: admissionv1.Create,
			wantDetail: "CREATE requires object and no oldObject (objectPresent=false, oldObjectPresent=false)"},
		{name: "create unexpected oldObject", operation: admissionv1.Create, object: true, oldObject: true,
			wantDetail: "CREATE requires object and no oldObject (objectPresent=true, oldObjectPresent=true)"},
		{name: "create missing object and unexpected oldObject", operation: admissionv1.Create, oldObject: true,
			wantDetail: "CREATE requires object and no oldObject (objectPresent=false, oldObjectPresent=true)"},
		{name: "update valid", operation: admissionv1.Update, object: true, oldObject: true},
		{name: "update missing both", operation: admissionv1.Update,
			wantDetail: "UPDATE requires object and oldObject (objectPresent=false, oldObjectPresent=false)"},
		{name: "update missing object", operation: admissionv1.Update, oldObject: true,
			wantDetail: "UPDATE requires object and oldObject (objectPresent=false, oldObjectPresent=true)"},
		{name: "update missing oldObject", operation: admissionv1.Update, object: true,
			wantDetail: "UPDATE requires object and oldObject (objectPresent=true, oldObjectPresent=false)"},
		{name: "delete valid", operation: admissionv1.Delete, oldObject: true},
		{name: "delete missing oldObject", operation: admissionv1.Delete,
			wantDetail: "DELETE requires oldObject and no object (objectPresent=false, oldObjectPresent=false)"},
		{name: "delete unexpected object", operation: admissionv1.Delete, object: true, oldObject: true,
			wantDetail: "DELETE requires oldObject and no object (objectPresent=true, oldObjectPresent=true)"},
		{name: "delete missing oldObject and unexpected object", operation: admissionv1.Delete, object: true,
			wantDetail: "DELETE requires oldObject and no object (objectPresent=true, oldObjectPresent=false)"},
		{name: "connect without object", operation: admissionv1.Connect},
		{name: "connect with object", operation: admissionv1.Connect, object: true},
		{name: "connect unexpected oldObject", operation: admissionv1.Connect, oldObject: true,
			wantDetail: "CONNECT must not carry oldObject (objectPresent=false, oldObjectPresent=true)"},
		{name: "connect both objects", operation: admissionv1.Connect, object: true, oldObject: true,
			wantDetail: "CONNECT must not carry oldObject (objectPresent=true, oldObjectPresent=true)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var output bytes.Buffer
			logger := textlogger.NewLogger(textlogger.NewConfig(textlogger.Verbosity(2), textlogger.Output(&output)))
			ctx := klog.NewContext(context.Background(), logger)
			req := &admissionv1.AdmissionRequest{
				UID:       "diagnostic-request-uid",
				Kind:      metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
				Resource:  metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
				Operation: tc.operation,
				Name:      "private-resource-name",
				Namespace: "private-namespace",
				UserInfo: authenticationv1.UserInfo{
					Username: "private-user-name",
					UID:      "private-user-uid",
				},
				Options: runtime.RawExtension{Raw: []byte(`{"credential":"private-option-credential"}`)},
			}
			object := runtime.RawExtension{Raw: []byte(`{"apiVersion":"apps/v1","kind":"Deployment","spec":{"credential":"private-payload-credential"}}`)}
			if tc.object {
				req.Object = object
			}
			if tc.oldObject {
				req.OldObject = object
			}
			objectPresent, oldObjectPresent, err := checkRequestObjectKinds(req)
			if err != nil || objectPresent != tc.object || oldObjectPresent != tc.oldObject {
				t.Fatalf("object-kind check = (%t, %t, %v), want (%t, %t, nil)", objectPresent, oldObjectPresent, err, tc.object, tc.oldObject)
			}
			gotDetail := ""
			if operationErr := checkOperationObjectPresence(req.Operation, objectPresent, oldObjectPresent); operationErr != nil {
				gotDetail = operationErr.Error()
			}
			if gotDetail != tc.wantDetail {
				t.Fatalf("operation check = %q, want %q", gotDetail, tc.wantDetail)
			}
			err = (&Verifier{}).checkDecodedRequest(ctx, req, []string{"apps"})
			if tc.wantDetail == "" {
				if err != nil || output.Len() != 0 {
					t.Fatalf("valid request: error=%v log=%q", err, output.String())
				}
				return
			}
			//nolint:errorlint // Wrapped errors must fail this identity assertion.
			if err != errVerificationFailed || err.Error() != genericDenyMessage {
				t.Fatalf("caller error = %v, want generic verification failure", err)
			}
			log := output.String()
			for _, want := range []string{
				`"Admission request rejected"`,
				`reason="invalid admission request"`,
				`requestUID="diagnostic-request-uid"`,
				"operation=" + strconv.Quote(string(tc.operation)),
				`resource="apps/v1, Resource=deployments"`,
				"detail=" + strconv.Quote(tc.wantDetail),
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
		})
	}
}
