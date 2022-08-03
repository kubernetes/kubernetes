/*
Copyright 2022 The Kubernetes Authors.

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
	"fmt"
	"net/http"

	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// CompressionDisabledFunc checks if a given request should disable compression.
type CompressionDisabledFunc func(*http.Request) (bool, error)

// WithCompressionDisabled stores result of CompressionDisabledFunc in context.
func WithCompressionDisabled(handler http.Handler, predicate CompressionDisabledFunc) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		decision, err := predicate(req)
		if err != nil {
			responsewriters.InternalError(w, req, fmt.Errorf("failed to determine if request should disable compression: %v", err))
			return
		}

		req = req.WithContext(request.WithCompressionDisabled(ctx, decision))
		handler.ServeHTTP(w, req)
	})
}
