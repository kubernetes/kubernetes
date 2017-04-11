/*
Copyright 2015 The Kubernetes Authors.

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

package http

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/probe"
)

const FailureCode int = -1

func TestHTTPProbeChecker(t *testing.T) {
	handleReq := func(s int, body string) func(w http.ResponseWriter, r *http.Request) {
		return func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(s)
			w.Write([]byte(body))
		}
	}

	prober := New()
	testCases := []struct {
		handler    func(w http.ResponseWriter, r *http.Request)
		reqHeaders http.Header
		health     probe.Result
		accBody    string
	}{
		// The probe will be filled in below.  This is primarily testing that an HTTP GET happens.
		{
			handler: handleReq(http.StatusOK, "ok body"),
			health:  probe.Success,
			accBody: "ok body",
		},
		{
			// Echo handler that returns the contents of request headers in the body
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(200)
				output := ""
				for k, arr := range r.Header {
					for _, v := range arr {
						output += fmt.Sprintf("%s: %s\n", k, v)
					}
				}
				w.Write([]byte(output))
			},
			reqHeaders: http.Header{
				"X-Muffins-Or-Cupcakes": {"muffins"},
			},
			health:  probe.Success,
			accBody: "X-Muffins-Or-Cupcakes: muffins",
		},
		{
			// Echo handler that returns the contents of Host in the body
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(200)
				w.Write([]byte(r.Host))
			},
			reqHeaders: http.Header{
				"Host": {"muffins.cupcakes.org"},
			},
			health:  probe.Success,
			accBody: "muffins.cupcakes.org",
		},
		{
			handler: handleReq(FailureCode, "fail body"),
			health:  probe.Failure,
		},
		{
			handler: handleReq(http.StatusInternalServerError, "fail body"),
			health:  probe.Failure,
		},
		{
			handler: func(w http.ResponseWriter, r *http.Request) {
				time.Sleep(3 * time.Second)
			},
			health: probe.Failure,
		},
	}
	for i, test := range testCases {
		func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				test.handler(w, r)
			}))
			defer server.Close()
			u, err := url.Parse(server.URL)
			if err != nil {
				t.Errorf("case %d: unexpected error: %v", i, err)
			}
			_, port, err := net.SplitHostPort(u.Host)
			if err != nil {
				t.Errorf("case %d: unexpected error: %v", i, err)
			}
			_, err = strconv.Atoi(port)
			if err != nil {
				t.Errorf("case %d: unexpected error: %v", i, err)
			}
			health, output, err := prober.Probe(u, test.reqHeaders, 1*time.Second)
			if test.health == probe.Unknown && err == nil {
				t.Errorf("case %d: expected error", i)
			}
			if test.health != probe.Unknown && err != nil {
				t.Errorf("case %d: unexpected error: %v", i, err)
			}
			if health != test.health {
				t.Errorf("case %d: expected %v, got %v", i, test.health, health)
			}
			if health != probe.Failure && test.health != probe.Failure {
				if !strings.Contains(output, test.accBody) {
					t.Errorf("Expected response body to contain %v, got %v", test.accBody, output)
				}
			}
		}()
	}
}
