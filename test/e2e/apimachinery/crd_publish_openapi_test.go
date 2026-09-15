/*
Copyright 2026 The Kubernetes Authors.

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

package apimachinery

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"k8s.io/kube-openapi/pkg/validation/spec"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestWaitForOpenAPISchemaRetriesRequestErrors(t *testing.T) {
	requestCount := 0
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		requestCount++
		if requestCount == 1 {
			return nil, errors.New("temporary network failure")
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{},
			Body:       io.NopCloser(strings.NewReader(`{"swagger":"2.0"}`)),
			Request:    req,
		}, nil
	})}

	err := waitForOpenAPISchemaWithClient(client, "https://example.com/openapi/v2", time.Millisecond, time.Second, func(*spec.Swagger) (bool, string) {
		return true, ""
	})
	if err != nil {
		t.Fatalf("waitForOpenAPISchemaWithClient returned an error: %v", err)
	}
	if want := waitSuccessThreshold + 1; requestCount != want {
		t.Fatalf("expected %d requests, got %d", want, requestCount)
	}
}
