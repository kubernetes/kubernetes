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

package openstack

import (
	"math/rand"
	"net/http"
	"testing"
)

type testRoundTripper struct{}

func (trt *testRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Header.Get("Authorization") == "Bearer " {
		return &http.Response{
			StatusCode: http.StatusUnauthorized,
		}, nil
	}
	return &http.Response{StatusCode: http.StatusOK}, nil
}

const LetterBytes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

func RandStringBytes(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = LetterBytes[rand.Intn(len(LetterBytes))]
	}
	return string(b)
}

func testGetToken() (string, error) {
	return RandStringBytes(32), nil
}

func TestOpenstackAuthProvider(t *testing.T) {
	trt := &tokenRoundTripper{
		RoundTripper: &testRoundTripper{},
		refreshFunc:  testGetToken,
	}
	// The first time, it will populate a token
	req, err := http.NewRequest(http.MethodGet, "https://example.com", nil)
	if err != nil {
		t.Errorf("failed to new request first round: %s", err)
	}
	resp, err := trt.RoundTrip(req)
	if err != nil {
		t.Errorf("unexpected error when round trip request first round: %s", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("unexpected status code first round: %d, expect %d", resp.StatusCode, http.StatusOK)
	}
	token := trt.token
	if token == "" {
		t.Errorf("expect to see populated token, but is empty")
	}
	// do once again, ensure it still works the second run
	resp, err = trt.RoundTrip(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Errorf("it should still work at next request, but see err(%s) or statuscode(%d)", err, resp.StatusCode)
	}
	// reset the token to be empty, and hope to see another refreshed token
	trt.token = ""
	resp, err = trt.RoundTrip(req)
	if err != nil {
		t.Errorf("unexpected error when round trip request the second round: %s", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("unexpected status code the second round: %d, expect %d", resp.StatusCode, http.StatusOK)
	}
	if token == trt.token || trt.token == "" {
		t.Errorf("unexpected token %q, should be different with %q", trt.token, token)
	}
}
