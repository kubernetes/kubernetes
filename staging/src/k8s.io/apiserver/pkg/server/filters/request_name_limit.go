/*
Copyright The Kubernetes Authors.

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

	apirequest "k8s.io/apiserver/pkg/endpoints/request"
)

// This rejects resource requests whose resourec name exceeds maxLen.
// // This prevents big user given names from reaching request
// paths, like metrics/error handling.
func WithResourceNameLengthLimit(handler http.Handler, maxLen int) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		info, ok := apirequest.RequestInfoFrom(req.Context())
		if !ok || info == nil || !info.IsResourceRequest {
			handler.ServeHTTP(w, req)
			return
		}

		if len(info.Name) > maxLen {
			http.Error(w, "resource name too long", http.StatusBadRequest)
			return
		}

		handler.ServeHTTP(w, req)
	})
}
