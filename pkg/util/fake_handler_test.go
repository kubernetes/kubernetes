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

package util

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestFakeHandlerPath(t *testing.T) {
	handler := FakeHandler{}
	server := httptest.NewServer(&handler)
	defer server.Close()
	method := "GET"
	path := "/foo/bar"
	body := "somebody"

	req, err := http.NewRequest(method, server.URL+path, bytes.NewBufferString(body))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	client := http.Client{}
	_, err = client.Do(req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	handler.ValidateRequest(t, path, method, &body)
}

func TestFakeHandlerPathNoBody(t *testing.T) {
	handler := FakeHandler{}
	server := httptest.NewServer(&handler)
	defer server.Close()
	method := "GET"
	path := "/foo/bar"

	req, err := http.NewRequest(method, server.URL+path, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	client := http.Client{}
	_, err = client.Do(req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	handler.ValidateRequest(t, path, method, nil)
}

type fakeError struct {
	errors []string
}

func (f *fakeError) Errorf(format string, args ...interface{}) {
	f.errors = append(f.errors, format)
}

func (f *fakeError) Logf(format string, args ...interface{}) {}

func TestFakeHandlerWrongPath(t *testing.T) {
	handler := FakeHandler{}
	server := httptest.NewServer(&handler)
	defer server.Close()
	method := "GET"
	path := "/foo/bar"
	fakeT := fakeError{}

	req, err := http.NewRequest(method, server.URL+"/foo/baz", nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	client := http.Client{}
	_, err = client.Do(req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	handler.ValidateRequest(&fakeT, path, method, nil)
	if len(fakeT.errors) != 1 {
		t.Errorf("Unexpected error set: %#v", fakeT.errors)
	}
}

func TestFakeHandlerWrongMethod(t *testing.T) {
	handler := FakeHandler{}
	server := httptest.NewServer(&handler)
	defer server.Close()
	method := "GET"
	path := "/foo/bar"
	fakeT := fakeError{}

	req, err := http.NewRequest("PUT", server.URL+path, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	client := http.Client{}
	_, err = client.Do(req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	handler.ValidateRequest(&fakeT, path, method, nil)
	if len(fakeT.errors) != 1 {
		t.Errorf("Unexpected error set: %#v", fakeT.errors)
	}
}

func TestFakeHandlerWrongBody(t *testing.T) {
	handler := FakeHandler{}
	server := httptest.NewServer(&handler)
	defer server.Close()
	method := "GET"
	path := "/foo/bar"
	body := "somebody"
	fakeT := fakeError{}

	req, err := http.NewRequest(method, server.URL+path, bytes.NewBufferString(body))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	client := http.Client{}
	_, err = client.Do(req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	otherbody := "otherbody"
	handler.ValidateRequest(&fakeT, path, method, &otherbody)
	if len(fakeT.errors) != 1 {
		t.Errorf("Unexpected error set: %#v", fakeT.errors)
	}
}

func TestFakeHandlerNilBody(t *testing.T) {
	handler := FakeHandler{}
	server := httptest.NewServer(&handler)
	defer server.Close()
	method := "GET"
	path := "/foo/bar"
	body := "somebody"
	fakeT := fakeError{}

	req, err := http.NewRequest(method, server.URL+path, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	client := http.Client{}
	_, err = client.Do(req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	handler.ValidateRequest(&fakeT, path, method, &body)
	if len(fakeT.errors) != 1 {
		t.Errorf("Unexpected error set: %#v", fakeT.errors)
	}
}
