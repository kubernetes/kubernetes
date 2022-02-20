/*
Copyright 2021 The Kubernetes Authors.

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

	"k8s.io/apiserver/pkg/endpoints/request"
)

// WithLatencyTrackers adds a LatencyTrackers instance to the
// context associated with a request so that we can measure latency
// incurred in various components within the apiserver.
func WithLatencyTrackers(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		req = req.WithContext(request.WithLatencyTrackers(ctx))
		handler.ServeHTTP(w, req)
	})
}
