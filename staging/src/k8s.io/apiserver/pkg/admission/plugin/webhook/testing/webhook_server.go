/*
Copyright 2018 The Kubernetes Authors.

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

package testing

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/api/admission/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/testcerts"
)

// NewTestServer returns a webhook test HTTPS server with fixed webhook test certs.
func NewTestServer(t *testing.T) *httptest.Server {
	// Create the test webhook server
	sCert, err := tls.X509KeyPair(testcerts.ServerCert, testcerts.ServerKey)
	if err != nil {
		t.Fatal(err)
	}
	rootCAs := x509.NewCertPool()
	rootCAs.AppendCertsFromPEM(testcerts.CACert)
	testServer := httptest.NewUnstartedServer(http.HandlerFunc(webhookHandler))
	testServer.TLS = &tls.Config{
		Certificates: []tls.Certificate{sCert},
		ClientCAs:    rootCAs,
		ClientAuth:   tls.RequireAndVerifyClientCert,
	}
	return testServer
}

func webhookHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Printf("got req: %v\n", r.URL.Path)
	switch r.URL.Path {
	case "/internalErr":
		http.Error(w, "webhook internal server error", http.StatusInternalServerError)
		return
	case "/invalidReq":
		w.WriteHeader(http.StatusSwitchingProtocols)
		w.Write([]byte("webhook invalid request"))
		return
	case "/invalidResp":
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("webhook invalid response"))
	case "/disallow":
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed: false,
				Result: &metav1.Status{
					Code: http.StatusForbidden,
				},
			},
		})
	case "/disallowReason":
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed: false,
				Result: &metav1.Status{
					Message: "you shall not pass",
					Code:    http.StatusForbidden,
				},
			},
		})
	case "/shouldNotBeCalled":
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed: false,
				Result: &metav1.Status{
					Message: "doesn't expect labels to match object selector",
					Code:    http.StatusForbidden,
				},
			},
		})
	case "/allow":
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed: true,
				AuditAnnotations: map[string]string{
					"key1": "value1",
				},
			},
		})
	case "/removeLabel":
		w.Header().Set("Content-Type", "application/json")
		pt := v1beta1.PatchTypeJSONPatch
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed:   true,
				PatchType: &pt,
				Patch:     []byte(`[{"op": "remove", "path": "/metadata/labels/remove"}]`),
				AuditAnnotations: map[string]string{
					"key1": "value1",
				},
			},
		})
	case "/addLabel":
		w.Header().Set("Content-Type", "application/json")
		pt := v1beta1.PatchTypeJSONPatch
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed:   true,
				PatchType: &pt,
				Patch:     []byte(`[{"op": "add", "path": "/metadata/labels/added", "value": "test"}]`),
			},
		})
	case "/invalidMutation":
		w.Header().Set("Content-Type", "application/json")
		pt := v1beta1.PatchTypeJSONPatch
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed:   true,
				PatchType: &pt,
				Patch:     []byte(`[{"op": "add", "CORRUPTED_KEY":}]`),
			},
		})
	case "/nilResponse":
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{})
	case "/invalidAnnotation":
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed: true,
				AuditAnnotations: map[string]string{
					"invalid*key": "value1",
				},
			},
		})
	case "/noop":
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed: true,
			},
		})
	default:
		http.NotFound(w, r)
	}
}
