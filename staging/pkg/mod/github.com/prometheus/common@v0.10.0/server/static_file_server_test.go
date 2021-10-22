// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package server

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

type dummyFileSystem struct{}

func (fs dummyFileSystem) Open(path string) (http.File, error) {
	return http.Dir(".").Open(".")
}

func TestServeHttp(t *testing.T) {
	cases := []struct {
		name        string
		path        string
		contentType string
	}{
		{
			name:        "normal file",
			path:        "index.html",
			contentType: "",
		},
		{
			name:        "javascript",
			path:        "test.js",
			contentType: "application/javascript",
		},
		{
			name:        "css",
			path:        "test.css",
			contentType: "text/css",
		},
		{
			name:        "png",
			path:        "test.png",
			contentType: "image/png",
		},
		{
			name:        "jpg",
			path:        "test.jpg",
			contentType: "image/jpeg",
		},
		{
			name:        "gif",
			path:        "test.gif",
			contentType: "image/gif",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rr := httptest.NewRecorder()
			req, err := http.NewRequest("GET", "http://localhost/"+c.path, nil)

			if err != nil {
				t.Fatal(err)
			}

			s := StaticFileServer(dummyFileSystem{})
			s.ServeHTTP(rr, req)

			if rr.Header().Get("Content-Type") != c.contentType {
				t.Fatalf("Unexpected Content-Type: %s", rr.Header().Get("Content-Type"))
			}
		})
	}
}
