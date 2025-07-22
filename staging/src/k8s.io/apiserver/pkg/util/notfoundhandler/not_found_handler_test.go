/*
Copyright 2021 The Kubernetes Authors.

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

package notfoundhandler

import (
	"context"
	"io"
	"net/http/httptest"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

func TestNotFoundHandler(t *testing.T) {
	isMuxAndDiscoveryCompleteGlobalValue := true
	isMuxAndDiscoveryCompleteTestFn := func(ctx context.Context) bool { return isMuxAndDiscoveryCompleteGlobalValue }
	serializer := serializer.NewCodecFactory(runtime.NewScheme()).WithoutConversion()
	target := New(serializer, isMuxAndDiscoveryCompleteTestFn)

	// scenario 1: pretend the request has been made after the signal has been ready
	req := httptest.NewRequest("GET", "http://apiserver.com/apis/flowcontrol.apiserver.k8s.io/v1beta1", nil)
	rw := httptest.NewRecorder()

	target.ServeHTTP(rw, req)
	resp := rw.Result()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	bodyStr := strings.TrimSuffix(string(body), "\n")

	if resp.StatusCode != 404 {
		t.Fatalf("unexpected status code %d, expected 503", resp.StatusCode)
	}
	expectedMsg := "404 page not found"
	if bodyStr != expectedMsg {
		t.Fatalf("unexpected response: %v, expected: %v", bodyStr, expectedMsg)
	}

	// scenario 2: pretend the request has been made before the signal has been ready
	isMuxAndDiscoveryCompleteGlobalValue = false
	rw = httptest.NewRecorder()

	target.ServeHTTP(rw, req)
	resp = rw.Result()
	body, err = io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	bodyStr = strings.TrimSuffix(string(body), "\n")
	if resp.StatusCode != 503 {
		t.Fatalf("unexpected status code %d, expected 503", resp.StatusCode)
	}
	expectedMsg = `{"kind":"Status","apiVersion":"v1","metadata":{},"status":"Failure","message":"the request has been made before all known HTTP paths have been installed, please try again","reason":"ServiceUnavailable","details":{"retryAfterSeconds":5},"code":503}`
	if bodyStr != expectedMsg {
		t.Fatalf("unexpected response: %v, expected: %v", bodyStr, expectedMsg)
	}
}
