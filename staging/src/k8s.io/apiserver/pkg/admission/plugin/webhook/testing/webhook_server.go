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
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"k8s.io/api/admission/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/testcerts"
	testingclock "k8s.io/utils/clock/testing"
)

// NewTestServerWithHandler returns a webhook test HTTPS server
// which uses given handler function to handle requests
func NewTestServerWithHandler(t testing.TB, handler func(http.ResponseWriter, *http.Request)) *httptest.Server {
	// Create the test webhook server
	sCert, err := tls.X509KeyPair(testcerts.ServerCert, testcerts.ServerKey)
	if err != nil {
		t.Error(err)
		t.FailNow()
	}
	rootCAs := x509.NewCertPool()
	rootCAs.AppendCertsFromPEM(testcerts.CACert)
	testServer := httptest.NewUnstartedServer(http.HandlerFunc(handler))
	testServer.TLS = &tls.Config{
		Certificates: []tls.Certificate{sCert},
		ClientCAs:    rootCAs,
		ClientAuth:   tls.RequireAndVerifyClientCert,
	}
	return testServer
}

// NewTestServer returns a webhook test HTTPS server with fixed webhook test certs.
func NewTestServer(t testing.TB) *httptest.Server {
	return NewTestServerWithHandler(t, webhookHandler)
}

func webhookHandler(w http.ResponseWriter, r *http.Request) {
	// fmt.Printf("got req: %v\n", r.URL.Path)
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
	case "/invalidPatch":
		w.Header().Set("Content-Type", "application/json")
		pt := v1beta1.PatchTypeJSONPatch
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed:   true,
				PatchType: &pt,
				Patch:     []byte(`[{`),
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
	case "/nonStatusError":
		hj, _ := w.(http.Hijacker)
		conn, _, _ := hj.Hijack()
		defer conn.Close()             //nolint:errcheck
		conn.Write([]byte("bad-http")) //nolint:errcheck
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

// ClockSteppingWebhookHandler given a fakeClock returns a request handler
// that moves time in given clock by an amount specified in the webhook request
func ClockSteppingWebhookHandler(t testing.TB, fakeClock *testingclock.FakeClock) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		validPath := regexp.MustCompile(`^/(?:allow|disallow)/(\d{1,10})$`)

		if !validPath.MatchString(path) {
			t.Errorf("error in test case, wrong webhook url path: '%q' expected to match: '%q'", path, validPath.String())
			t.FailNow()
		}

		delay, _ := strconv.ParseInt(validPath.FindStringSubmatch(path)[1], 0, 64)
		fakeClock.Step(time.Duration(delay))
		w.Header().Set("Content-Type", "application/json")

		if strings.HasPrefix(path, "/allow/") {
			json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
				Response: &v1beta1.AdmissionResponse{
					Allowed: true,
					AuditAnnotations: map[string]string{
						"key1": "value1",
					},
				},
			})
			return
		}
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed: false,
				Result: &metav1.Status{
					Code: http.StatusForbidden,
				},
			},
		})
	}
}
