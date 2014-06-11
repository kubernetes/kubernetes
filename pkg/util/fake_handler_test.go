package util

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
)

func expectNoError(t *testing.T, err error) {
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
}

func TestFakeHandlerPath(t *testing.T) {
	handler := FakeHandler{}
	server := httptest.NewServer(&handler)
	method := "GET"
	path := "/foo/bar"
	body := "somebody"

	req, err := http.NewRequest(method, server.URL+path, bytes.NewBufferString(body))
	expectNoError(t, err)
	client := http.Client{}
	_, err = client.Do(req)
	expectNoError(t, err)

	handler.ValidateRequest(t, path, method, &body)
}

func TestFakeHandlerPathNoBody(t *testing.T) {
	handler := FakeHandler{}
	server := httptest.NewServer(&handler)
	method := "GET"
	path := "/foo/bar"

	req, err := http.NewRequest(method, server.URL+path, nil)
	expectNoError(t, err)
	client := http.Client{}
	_, err = client.Do(req)
	expectNoError(t, err)

	handler.ValidateRequest(t, path, method, nil)
}

type fakeError struct {
	errors []string
}

func (f *fakeError) Errorf(format string, args ...interface{}) {
	f.errors = append(f.errors, format)
}

func TestFakeHandlerWrongPath(t *testing.T) {
	handler := FakeHandler{}
	server := httptest.NewServer(&handler)
	method := "GET"
	path := "/foo/bar"
	fakeT := fakeError{}

	req, err := http.NewRequest(method, server.URL+"/foo/baz", nil)
	expectNoError(t, err)
	client := http.Client{}
	_, err = client.Do(req)
	expectNoError(t, err)

	handler.ValidateRequest(&fakeT, path, method, nil)
	if len(fakeT.errors) != 1 {
		t.Errorf("Unexpected error set: %#v", fakeT.errors)
	}
}

func TestFakeHandlerWrongMethod(t *testing.T) {
	handler := FakeHandler{}
	server := httptest.NewServer(&handler)
	method := "GET"
	path := "/foo/bar"
	fakeT := fakeError{}

	req, err := http.NewRequest("PUT", server.URL+path, nil)
	expectNoError(t, err)
	client := http.Client{}
	_, err = client.Do(req)
	expectNoError(t, err)

	handler.ValidateRequest(&fakeT, path, method, nil)
	if len(fakeT.errors) != 1 {
		t.Errorf("Unexpected error set: %#v", fakeT.errors)
	}
}

func TestFakeHandlerWrongBody(t *testing.T) {
	handler := FakeHandler{}
	server := httptest.NewServer(&handler)
	method := "GET"
	path := "/foo/bar"
	body := "somebody"
	fakeT := fakeError{}

	req, err := http.NewRequest(method, server.URL+path, bytes.NewBufferString(body))
	expectNoError(t, err)
	client := http.Client{}
	_, err = client.Do(req)
	expectNoError(t, err)

	otherbody := "otherbody"
	handler.ValidateRequest(&fakeT, path, method, &otherbody)
	if len(fakeT.errors) != 1 {
		t.Errorf("Unexpected error set: %#v", fakeT.errors)
	}
}

func TestFakeHandlerNilBody(t *testing.T) {
	handler := FakeHandler{}
	server := httptest.NewServer(&handler)
	method := "GET"
	path := "/foo/bar"
	body := "somebody"
	fakeT := fakeError{}

	req, err := http.NewRequest(method, server.URL+path, nil)
	expectNoError(t, err)
	client := http.Client{}
	_, err = client.Do(req)
	expectNoError(t, err)

	handler.ValidateRequest(&fakeT, path, method, &body)
	if len(fakeT.errors) != 1 {
		t.Errorf("Unexpected error set: %#v", fakeT.errors)
	}
}
