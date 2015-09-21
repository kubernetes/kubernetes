/*
Copyright 2014 Google Inc. All rights reserved.

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

package apiserver

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

type fakeRL bool

func (fakeRL) Stop()             {}
func (f fakeRL) CanAccept() bool { return bool(f) }

func TestRateLimit(t *testing.T) {
	for _, allow := range []bool{true, false} {
		rl := fakeRL(allow)
		server := httptest.NewServer(RateLimit(rl, http.HandlerFunc(
			func(w http.ResponseWriter, req *http.Request) {
				if !allow {
					t.Errorf("Unexpected call")
				}
			},
		)))
		defer server.Close()
		http.Get(server.URL)
	}
}

func TestReadOnly(t *testing.T) {
	server := httptest.NewServer(ReadOnly(http.HandlerFunc(
		func(w http.ResponseWriter, req *http.Request) {
			if req.Method != "GET" {
				t.Errorf("Unexpected call: %v", req.Method)
			}
		},
	)))
	defer server.Close()
	for _, verb := range []string{"GET", "POST", "PUT", "DELETE", "CREATE"} {
		req, err := http.NewRequest(verb, server.URL, nil)
		if err != nil {
			t.Fatalf("Couldn't make request: %v", err)
		}
		http.DefaultClient.Do(req)
	}
}
