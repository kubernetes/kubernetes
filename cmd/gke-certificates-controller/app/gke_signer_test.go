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

package app

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"
	"text/template"
	"time"

	"k8s.io/client-go/tools/record"
	certificates "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
)

const kubeConfigTmpl = `
clusters:
- cluster:
    server: {{ .Server }}
    name: testcluster
users:
- user:
    username: admin
    password: mypass
`

func TestGKESigner(t *testing.T) {
	goodResponse := &certificates.CertificateSigningRequest{
		Status: certificates.CertificateSigningRequestStatus{
			Certificate: []byte("fake certificate"),
		},
	}

	invalidResponse := "{ \"status\": \"Not a properly formatted CSR response\" }"

	cases := []struct {
		mockResponse interface{}
		expected     []byte
		failCalls    int
		wantErr      bool
	}{
		{
			mockResponse: goodResponse,
			expected:     goodResponse.Status.Certificate,
			wantErr:      false,
		},
		{
			mockResponse: goodResponse,
			expected:     goodResponse.Status.Certificate,
			failCalls:    3,
			wantErr:      false,
		},
		{
			mockResponse: goodResponse,
			failCalls:    20,
			wantErr:      true,
		},
		{
			mockResponse: invalidResponse,
			wantErr:      true,
		},
	}

	for _, c := range cases {
		server, err := newTestServer(c.mockResponse, c.failCalls)
		if err != nil {
			t.Fatalf("error creating test server")
		}

		kubeConfig, err := ioutil.TempFile("", "kubeconfig")
		if err != nil {
			t.Fatalf("error creating kubeconfig tempfile: %v", err)
		}

		tmpl, err := template.New("kubeconfig").Parse(kubeConfigTmpl)
		if err != nil {
			t.Fatalf("error creating kubeconfig template: %v", err)
		}

		data := struct{ Server string }{server.httpserver.URL}

		if err := tmpl.Execute(kubeConfig, data); err != nil {
			t.Fatalf("error executing kubeconfig template: %v", err)
		}

		if err := kubeConfig.Close(); err != nil {
			t.Fatalf("error closing kubeconfig template: %v", err)
		}

		signer, err := NewGKESigner(kubeConfig.Name(), time.Duration(500)*time.Millisecond, record.NewFakeRecorder(10), nil)
		if err != nil {
			t.Fatalf("error creating GKESigner: %v", err)
		}

		cert, err := signer.sign(&certificates.CertificateSigningRequest{})

		if c.wantErr {
			if err == nil {
				t.Errorf("wanted error during GKE.Sign() call, got not none")
			}
		} else {
			if err != nil {
				t.Errorf("error while signing: %v", err)
			}

			if !bytes.Equal(cert.Status.Certificate, c.expected) {
				t.Errorf("response certificate didn't match expected %v: %v", c.expected, cert)
			}
		}
	}
}

type testServer struct {
	httpserver *httptest.Server
	failCalls  int
	response   interface{}
}

func newTestServer(response interface{}, failCalls int) (*testServer, error) {
	server := &testServer{
		response:  response,
		failCalls: failCalls,
	}

	server.httpserver = httptest.NewServer(server)
	return server, nil
}

func (s *testServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if s.failCalls > 0 {
		http.Error(w, "Service unavailable", 500)
		s.failCalls--
	} else {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(s.response)
	}
}
