/*
Copyright 2020 The Kubernetes Authors.

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
	"strings"
)

// WithHSTS is a simple HSTS implementation that wraps an http Handler.
// If hstsDirectives is empty or nil, no HSTS support is installed.
func WithHSTS(handler http.Handler, hstsDirectives []string) http.Handler {
	if len(hstsDirectives) == 0 {
		return handler
	}
	allDirectives := strings.Join(hstsDirectives, "; ")
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// Chrome and Mozilla Firefox maintain an HSTS preload list
		// issue : golang.org/issue/26162
		// Set the Strict-Transport-Security header if it is not already set
		if _, ok := w.Header()["Strict-Transport-Security"]; !ok {
			w.Header().Set("Strict-Transport-Security", allDirectives)
		}
		handler.ServeHTTP(w, req)
	})
}
