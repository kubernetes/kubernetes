/*
Copyright 2024 The Kubernetes Authors.

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

	"k8s.io/apiserver/pkg/header"
)

// WithSafeHeaderWriter adds an interface for setting response headers that is threadsafe and can be frozen
func WithSafeHeaderWriter(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := header.WithSafeHeaderWriter(r.Context(), w.Header())
		r = r.WithContext(ctx)
		handler.ServeHTTP(w, r)
	})
}
