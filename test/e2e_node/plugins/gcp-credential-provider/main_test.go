/*
Copyright 2022 The Kubernetes Authors.

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

package main

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

type fakeTokenServer struct {
	token string
}

func (f *fakeTokenServer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	fmt.Fprintf(w, `{"access_token": "%s"}`, f.token)
}

func Test_getCredentials(t *testing.T) {
	server := httptest.NewServer(&fakeTokenServer{token: "abc123"})
	defer server.Close()

	in := bytes.NewBuffer([]byte(`{"kind":"CredentialProviderRequest","apiVersion":"credentialprovider.kubelet.k8s.io/v1","image":"gcr.io/foobar"}`))
	out := bytes.NewBuffer(nil)

	err := getCredentials(server.URL, in, out)
	if err != nil {
		t.Fatalf("unexpected error running getCredentials: %v", err)
	}

	expected := `{"kind":"CredentialProviderResponse","apiVersion":"credentialprovider.kubelet.k8s.io/v1","cacheKeyType":"Registry","auth":{"*.gcr.io":{"username":"_token","password":"abc123"},"*.pkg.dev":{"username":"_token","password":"abc123"},"container.cloud.google.com":{"username":"_token","password":"abc123"},"gcr.io":{"username":"_token","password":"abc123"}}}
`

	if out.String() != expected {
		t.Logf("actual response: %v", out)
		t.Logf("expected response: %v", expected)
		t.Errorf("unexpected credential provider response")
	}
}
