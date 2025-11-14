/*
Copyright 2014 The Kubernetes Authors.

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

package rest

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestInputStreamReader(t *testing.T) {
	resultString := "Test output"
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Write([]byte(resultString))
	}))
	defer s.Close()
	u, err := url.Parse(s.URL)
	if err != nil {
		t.Errorf("Error parsing server URL: %v", err)
		return
	}
	streamer := &LocationStreamer{
		Location: u,
	}
	readCloser, _, _, err := streamer.InputStream(context.Background(), "", "")
	if err != nil {
		t.Errorf("Unexpected error when getting stream: %v", err)
		return
	}
	defer readCloser.Close()
	result, _ := io.ReadAll(readCloser)
	if string(result) != resultString {
		t.Errorf("Stream content does not match. Got: %s. Expected: %s.", string(result), resultString)
	}
}

func TestInputStreamNullLocation(t *testing.T) {
	streamer := &LocationStreamer{
		Location: nil,
	}
	readCloser, _, _, err := streamer.InputStream(context.Background(), "", "")
	if err != nil {
		t.Errorf("Unexpected error when getting stream with null location: %v", err)
	}
	if readCloser != nil {
		t.Errorf("Expected stream to be nil. Got: %#v", readCloser)
	}
}

type testTransport struct {
	body string
}

func (tt *testTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	r := bufio.NewReader(bytes.NewBufferString(tt.body))
	return http.ReadResponse(r, req)
}

func fakeTransport(mime, message string) http.RoundTripper {
	content := fmt.Sprintf("HTTP/1.1 200 OK\nContent-Type: %s\n\n%s", mime, message)
	return &testTransport{body: content}
}

func TestInputStreamContentType(t *testing.T) {
	location, _ := url.Parse("http://www.example.com")
	streamer := &LocationStreamer{
		Location:  location,
		Transport: fakeTransport("application/json", "hello world"),
	}
	readCloser, _, contentType, err := streamer.InputStream(context.Background(), "", "")
	if err != nil {
		t.Errorf("Unexpected error when getting stream: %v", err)
		return
	}
	defer readCloser.Close()
	if contentType != "application/json" {
		t.Errorf("Unexpected content type. Got: %s. Expected: application/json", contentType)
	}
}

func TestInputStreamTransport(t *testing.T) {
	message := "hello world"
	location, _ := url.Parse("http://www.example.com")
	streamer := &LocationStreamer{
		Location:  location,
		Transport: fakeTransport("text/plain", message),
	}
	readCloser, _, _, err := streamer.InputStream(context.Background(), "", "")
	if err != nil {
		t.Errorf("Unexpected error when getting stream: %v", err)
		return
	}
	defer readCloser.Close()
	result, _ := io.ReadAll(readCloser)
	if string(result) != message {
		t.Errorf("Stream content does not match. Got: %s. Expected: %s.", string(result), message)
	}
}

func fakeInternalServerErrorTransport(mime, message string) http.RoundTripper {
	content := fmt.Sprintf("HTTP/1.1 500 \"Internal Server Error\"\nContent-Type: %s\n\n%s", mime, message)
	return &testTransport{body: content}
}

func TestInputStreamInternalServerErrorTransport(t *testing.T) {
	message := "Pod is in PodPending"
	location, _ := url.Parse("http://www.example.com")
	streamer := &LocationStreamer{
		Location:        location,
		Transport:       fakeInternalServerErrorTransport("text/plain", message),
		ResponseChecker: NewGenericHttpResponseChecker(schema.GroupResource{}, ""),
	}
	expectedError := errors.NewInternalError(fmt.Errorf("%s", message))

	_, _, _, err := streamer.InputStream(context.Background(), "", "")
	if err == nil {
		t.Errorf("unexpected non-error")
		return
	}

	if !reflect.DeepEqual(err, expectedError) {
		t.Errorf("StreamInternalServerError does not match. Got: %s. Expected: %s.", err, expectedError)
	}
}

func TestInputStreamRedirects(t *testing.T) {
	const redirectPath = "/redirect"
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path == redirectPath {
			t.Fatal("Redirects should not be followed")
		} else {
			http.Redirect(w, req, redirectPath, http.StatusFound)
		}
	}))
	loc, err := url.Parse(s.URL)
	require.NoError(t, err, "Error parsing server URL")

	streamer := &LocationStreamer{
		Location:        loc,
		RedirectChecker: PreventRedirects,
	}
	_, _, _, err = streamer.InputStream(context.Background(), "", "")
	assert.Error(t, err, "Redirect should trigger an error")
}

func TestInputStreamContentTypeCharset(t *testing.T) {
	tests := []struct {
		name               string
		serverContentType  string // What the backend server sends
		expectedResultType string // What we expect after processing
	}{
		{
			name:               "text/plain without charset should add utf-8",
			serverContentType:  "text/plain",
			expectedResultType: "text/plain; charset=utf-8",
		},
		{
			name:               "text/plain with utf-8 should preserve it",
			serverContentType:  "text/plain; charset=utf-8",
			expectedResultType: "text/plain; charset=utf-8",
		},
		{
			name:               "text/plain with latin-1 should preserve it",
			serverContentType:  "text/plain; charset=latin-1",
			expectedResultType: "text/plain; charset=latin-1",
		},
		{
			name:               "text/plain with iso-8859-1 should preserve it",
			serverContentType:  "text/plain; charset=iso-8859-1",
			expectedResultType: "text/plain; charset=iso-8859-1",
		},
		{
			name:               "text/plain with extra spaces should preserve charset",
			serverContentType:  "text/plain;  charset=utf-16",
			expectedResultType: "text/plain;  charset=utf-16",
		},
		{
			name:               "application/json should not be modified",
			serverContentType:  "application/json",
			expectedResultType: "application/json",
		},
		{
			name:               "application/octet-stream should not be modified",
			serverContentType:  "application/octet-stream",
			expectedResultType: "application/octet-stream",
		},
		{
			name:               "empty content-type should default to text/plain with utf-8",
			serverContentType:  "",
			expectedResultType: "text/plain; charset=utf-8",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			location, _ := url.Parse("http://www.example.com")
			streamer := &LocationStreamer{
				Location:  location,
				Transport: fakeTransport(tt.serverContentType, "test content"),
			}
			_, _, contentType, err := streamer.InputStream(context.Background(), "", "")
			if err != nil {
				t.Errorf("Unexpected error when getting stream: %v", err)
				return
			}
			if contentType != tt.expectedResultType {
				t.Errorf("Content type mismatch.\nGot:      %q\nExpected: %q", contentType, tt.expectedResultType)
			}
		})
	}
}

func TestInputStreamContentTypePreset(t *testing.T) {
	// Test that if ContentType is preset in LocationStreamer, it's used unconditionally
	location, _ := url.Parse("http://www.example.com")
	streamer := &LocationStreamer{
		Location:    location,
		Transport:   fakeTransport("text/plain", "test content"),
		ContentType: "application/custom; charset=special", // Preset value
	}
	_, _, contentType, err := streamer.InputStream(context.Background(), "", "")
	if err != nil {
		t.Errorf("Unexpected error when getting stream: %v", err)
		return
	}
	if contentType != "application/custom; charset=special" {
		t.Errorf("Expected preset ContentType to be used. Got: %s", contentType)
	}
}
