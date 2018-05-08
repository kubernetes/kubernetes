/*
Copyright 2018 The Kubernetes Authors.

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

package downloader

import (
	"fmt"
	"net/http"
	"testing"
)

// TestHandlerDownloader creates a new SpecDownloader that points at fake openapi handler and
// tests that it can get a proto format spec for various etags
func TestHandlerDownloader(t *testing.T) {
	fakePB := "AAAAAAAAAAAAA"
	handler := &fakeHandler{
		swaggerPB: []byte(fakePB),
		etag:      "B",
	}

	testCases := []struct {
		// Inputs
		handler  http.Handler
		lastEtag string

		// Outputs
		expectedSpecBytes  string
		expectedNewEtag    string
		expectedHTTPStatus int
		expectedErr        bool
	}{
		{
			handler:            handler,
			lastEtag:           "",
			expectedSpecBytes:  fakePB,
			expectedNewEtag:    "B",
			expectedHTTPStatus: http.StatusOK,
		}, {
			handler:            handler,
			lastEtag:           "B",
			expectedNewEtag:    "B",
			expectedHTTPStatus: http.StatusNotModified,
		}, {
			handler:            handler,
			lastEtag:           "C",
			expectedSpecBytes:  fakePB,
			expectedNewEtag:    "B",
			expectedHTTPStatus: http.StatusOK,
		}, {
			handler:     &brokenHandler{},
			expectedErr: true,
		},
	}

	for _, tc := range testCases {
		downloader := NewHandlerDownloader(tc.handler, "/test/path", "application/test_content_type")
		specBytes, newEtag, httpStatus, err := downloader.Download(tc.lastEtag)
		if string(specBytes) != tc.expectedSpecBytes {
			t.Errorf("Expected specBytes to be %v but got %v", tc.expectedSpecBytes, string(specBytes))
		}
		if newEtag != tc.expectedNewEtag {
			t.Errorf("Expected newEtag to be %v but got %v", tc.expectedNewEtag, newEtag)
		}
		if httpStatus != tc.expectedHTTPStatus {
			t.Errorf("Expected httpStatus to be %v but got %v", tc.expectedHTTPStatus, httpStatus)
		}
		if (err != nil) && !tc.expectedErr {
			t.Errorf("Expected no error but got %v", err)
		}
		if (err == nil) && tc.expectedErr {
			t.Errorf("Expected error but got none")
		}
	}
}

type fakeHandler struct {
	swaggerPB []byte
	etag      string
}

var _ http.Handler = &brokenHandler{}

// ServeHTTP implements http.Handler
func (f *fakeHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if len(f.etag) > 0 {
		w.Header().Add("Etag", f.etag)
	}
	ifNoneMatches := r.Header["If-None-Match"]
	for _, match := range ifNoneMatches {
		if match == f.etag {
			w.WriteHeader(http.StatusNotModified)
			return
		}
	}

	accept := r.Header.Get("Accept")
	switch accept {
	case "application/test_content_type":
		w.Write(f.swaggerPB)
		w.Header().Set("Etag", f.etag)
	default:
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprintf(w, "Reqested format not implemented: %v\n", accept)
	}
}

type brokenHandler struct{}

var _ http.Handler = &brokenHandler{}

// ServeHTTP implements http.Handler
func (*brokenHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusNotFound)
}
