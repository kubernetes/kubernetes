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

package filters

import (
	"bytes"
	"compress/gzip"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/endpoints/filters"
	"k8s.io/apiserver/pkg/endpoints/request"
)

func TestCompression(t *testing.T) {
	tests := []struct {
		encoding string
		watch    bool
	}{
		{"", false},
		{"gzip", true},
		{"gzip", false},
	}

	responseData := []byte("1234")

	requestContextMapper := request.NewRequestContextMapper()

	for _, test := range tests {
		handler := WithCompression(
			http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				w.Write(responseData)
			}),
			requestContextMapper,
		)
		handler = filters.WithRequestInfo(handler, newTestRequestInfoResolver(), requestContextMapper)
		handler = request.WithRequestContext(handler, requestContextMapper)
		server := httptest.NewServer(handler)
		defer server.Close()
		client := http.Client{
			Transport: &http.Transport{
				DisableCompression: true,
			},
		}

		url := server.URL + "/api/v1/pods"
		if test.watch {
			url = url + "?watch=1"
		}
		request, err := http.NewRequest("GET", url, nil)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		request.Header.Set("Accept-Encoding", test.encoding)
		response, err := client.Do(request)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		var reader io.Reader
		if test.encoding == "gzip" && !test.watch {
			if response.Header.Get("Content-Encoding") != "gzip" {
				t.Fatal("expected response header Content-Encoding to be set to \"gzip\"")
			}
			if response.Header.Get("Vary") != "Accept-Encoding" {
				t.Fatal("expected response header Vary to be set to \"Accept-Encoding\"")
			}
			reader, err = gzip.NewReader(response.Body)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		} else {
			if response.Header.Get("Content-Encoding") == "gzip" {
				t.Fatal("expected response header Content-Encoding not to be set")
			}
			reader = response.Body
		}
		body, err := ioutil.ReadAll(reader)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !bytes.Equal(body, responseData) {
			t.Fatalf("Expected response body %s to equal %s", body, responseData)
		}
	}
}

func newTestRequestInfoResolver() *request.RequestInfoFactory {
	return &request.RequestInfoFactory{
		APIPrefixes:          sets.NewString("api", "apis"),
		GrouplessAPIPrefixes: sets.NewString("api"),
	}
}
