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
)

// WithCacheControl sets the Cache-Control header to "no-cache, private" because all servers are protected by authn/authz.
// see https://developers.google.com/web/fundamentals/performance/optimizing-content-efficiency/http-caching#defining_optimal_cache-control_policy
func WithCacheControl(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// Set the cache-control header if it is not already set
		if _, ok := w.Header()["Cache-Control"]; !ok {
			w.Header().Set("Cache-Control", "no-cache, private")
		}
		handler.ServeHTTP(w, req)
	})
}
