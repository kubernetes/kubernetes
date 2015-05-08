/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package client

import (
	"encoding/base64"
	"net/http"
	"testing"
)

func TestUnsecuredTLSTransport(t *testing.T) {
	cfg := NewUnsafeTLSConfig()
	if !cfg.InsecureSkipVerify {
		t.Errorf("expected config to be insecure")
	}
}

type testRoundTripper struct {
	Request  *http.Request
	Response *http.Response
	Err      error
}

func (rt *testRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.Request = req
	return rt.Response, rt.Err
}

func TestBearerAuthRoundTripper(t *testing.T) {
	rt := &testRoundTripper{}
	req := &http.Request{}
	NewBearerAuthRoundTripper("test", rt).RoundTrip(req)
	if rt.Request == nil {
		t.Fatalf("unexpected nil request: %v", rt)
	}
	if rt.Request == req {
		t.Fatalf("round tripper should have copied request object: %#v", rt.Request)
	}
	if rt.Request.Header.Get("Authorization") != "Bearer test" {
		t.Errorf("unexpected authorization header: %#v", rt.Request)
	}
}

func TestBasicAuthRoundTripper(t *testing.T) {
	rt := &testRoundTripper{}
	req := &http.Request{}
	NewBasicAuthRoundTripper("user", "pass", rt).RoundTrip(req)
	if rt.Request == nil {
		t.Fatalf("unexpected nil request: %v", rt)
	}
	if rt.Request == req {
		t.Fatalf("round tripper should have copied request object: %#v", rt.Request)
	}
	if rt.Request.Header.Get("Authorization") != "Basic "+base64.StdEncoding.EncodeToString([]byte("user:pass")) {
		t.Errorf("unexpected authorization header: %#v", rt.Request)
	}
}

func TestUserAgentRoundTripper(t *testing.T) {
	rt := &testRoundTripper{}
	req := &http.Request{
		Header: make(http.Header),
	}
	req.Header.Set("User-Agent", "other")
	NewUserAgentRoundTripper("test", rt).RoundTrip(req)
	if rt.Request == nil {
		t.Fatalf("unexpected nil request: %v", rt)
	}
	if rt.Request != req {
		t.Fatalf("round tripper should not have copied request object: %#v", rt.Request)
	}
	if rt.Request.Header.Get("User-Agent") != "other" {
		t.Errorf("unexpected user agent header: %#v", rt.Request)
	}

	req = &http.Request{}
	NewUserAgentRoundTripper("test", rt).RoundTrip(req)
	if rt.Request == nil {
		t.Fatalf("unexpected nil request: %v", rt)
	}
	if rt.Request == req {
		t.Fatalf("round tripper should have copied request object: %#v", rt.Request)
	}
	if rt.Request.Header.Get("User-Agent") != "test" {
		t.Errorf("unexpected user agent header: %#v", rt.Request)
	}
}
