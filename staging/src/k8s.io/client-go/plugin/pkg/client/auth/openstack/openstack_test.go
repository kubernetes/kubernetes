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
	"time"
)

// testTokenGetter is a simple random token getter.
type testTokenGetter struct{}

const LetterBytes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

func RandStringBytes(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = LetterBytes[rand.Intn(len(LetterBytes))]
	}
	return string(b)
}

func (*testTokenGetter) Token() (string, error) {
	return RandStringBytes(32), nil
}

// testRoundTripper is mocked roundtripper which responds with unauthorized when
// there is no authorization header, otherwise returns status ok.
type testRoundTripper struct{}

func (trt *testRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	authHeader := req.Header.Get("Authorization")
	if authHeader == "" || authHeader == "Bearer " {
		return &http.Response{
			StatusCode: http.StatusUnauthorized,
		}, nil
	}
	return &http.Response{StatusCode: http.StatusOK}, nil
}

func TestOpenstackAuthProvider(t *testing.T) {
	trt := &tokenRoundTripper{
		RoundTripper: &testRoundTripper{},
	}

	tests := []struct {
		name     string
		ttl      time.Duration
		interval time.Duration
		same     bool
	}{
		{
			name:     "normal",
			ttl:      2 * time.Second,
			interval: 1 * time.Second,
			same:     true,
		},
		{
			name:     "expire",
			ttl:      1 * time.Second,
			interval: 2 * time.Second,
			same:     false,
		},
	}

	for _, test := range tests {
		trt.tokenGetter = &cachedGetter{
			tokenGetter: &testTokenGetter{},
			ttl:         test.ttl,
		}

		req, err := http.NewRequest(http.MethodPost, "https://test-api-server.com", nil)
		if err != nil {
			t.Errorf("failed to new request: %s", err)
		}
		trt.RoundTrip(req)
		header := req.Header.Get("Authorization")
		if header == "" {
			t.Errorf("expect to see token in header, but is absent")
		}

		time.Sleep(test.interval)

		req, err = http.NewRequest(http.MethodPost, "https://test-api-server.com", nil)
		if err != nil {
			t.Errorf("failed to new request: %s", err)
		}
		trt.RoundTrip(req)
		newHeader := req.Header.Get("Authorization")
		if newHeader == "" {
			t.Errorf("expect to see token in header, but is absent")
		}

		same := newHeader == header
		if same != test.same {
			t.Errorf("expect to get %t when compare header, but saw %t", test.same, same)
		}
	}

}
