/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package chaosclient

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

type TestLogChaos struct {
	*testing.T
}

func (t TestLogChaos) OnChaos(req *http.Request, c Chaos) {
	t.Logf("CHAOS: chaotic behavior for %s %s: %v", req.Method, req.URL.String(), c)
}

func unwrapURLError(err error) error {
	if urlErr, ok := err.(*url.Error); ok && urlErr != nil {
		return urlErr.Err
	}
	return err
}

func TestChaos(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	client := http.Client{
		Transport: NewChaosRoundTripper(http.DefaultTransport, TestLogChaos{t}, ErrSimulatedConnectionResetByPeer),
	}
	resp, err := client.Get(server.URL)
	if unwrapURLError(err) != ErrSimulatedConnectionResetByPeer.error {
		t.Fatalf("expected reset by peer: %v", err)
	}
	if resp != nil {
		t.Fatalf("expected no response object: %#v", resp)
	}
}

func TestPartialChaos(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	seed := NewSeed(1)
	client := http.Client{
		Transport: NewChaosRoundTripper(
			http.DefaultTransport, TestLogChaos{t},
			seed.P(0.5, ErrSimulatedConnectionResetByPeer),
		),
	}
	success, fail := 0, 0
	for {
		_, err := client.Get(server.URL)
		if err != nil {
			fail++
		} else {
			success++
		}
		if success > 1 && fail > 1 {
			break
		}
	}
}
