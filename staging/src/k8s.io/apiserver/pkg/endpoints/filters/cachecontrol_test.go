/*
Copyright 2019 The Kubernetes Authors.

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

package filters

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCacheControl(t *testing.T) {
	tests := []struct {
		name string
		path string

		startingHeader string
		expectedHeader string
	}{
		{
			name:           "simple",
			path:           "/api/v1/namespaces",
			expectedHeader: "no-cache, private",
		},
		{
			name:           "openapi",
			path:           "/openapi/v2",
			expectedHeader: "no-cache, private",
		},
		{
			name:           "already-set",
			path:           "/api/v1/namespaces",
			startingHeader: "nonsense",
			expectedHeader: "nonsense",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			handler := http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				//do nothing
			})
			wrapped := WithCacheControl(handler)

			testRequest, err := http.NewRequest(http.MethodGet, test.path, nil)
			if err != nil {
				t.Fatal(err)
			}
			w := httptest.NewRecorder()
			if len(test.startingHeader) > 0 {
				w.Header().Set("Cache-Control", test.startingHeader)
			}

			wrapped.ServeHTTP(w, testRequest)
			actual := w.Header().Get("Cache-Control")

			if actual != test.expectedHeader {
				t.Fatal(actual)
			}
		})
	}

}
