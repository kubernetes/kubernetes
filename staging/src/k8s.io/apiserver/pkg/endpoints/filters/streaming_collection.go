/*
Copyright 2026 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// WithStreamingCollectionEncoding records whether a list response can be
// streamed, so APF can cap the memory estimate.
//
// This runs before routing, so it uses a server-wide serializer and makes a
// conservative prediction.
func WithStreamingCollectionEncoding(handler http.Handler, serializer runtime.NegotiatedSerializer) http.Handler {
	if serializer == nil {
		return handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		if requestInfo, ok := request.RequestInfoFrom(ctx); ok &&
			requestInfo.IsResourceRequest && requestInfo.Verb == "list" &&
			negotiatesStreamingCollectionEncoding(req, serializer) {
			req = req.WithContext(request.WithStreamingCollectionEncoding(ctx, true))
		}
		handler.ServeHTTP(w, req)
	})
}

// negotiatesStreamingCollectionEncoding reports whether the serializer selected
// for this request streams collection items individually.
func negotiatesStreamingCollectionEncoding(req *http.Request, serializer runtime.NegotiatedSerializer) bool {
	_, info, err := negotiation.NegotiateOutputMediaType(req, serializer, negotiation.DefaultEndpointRestrictions)
	if err != nil {
		return false
	}
	encoder, ok := info.Serializer.(runtime.StreamingCollectionEncoder)
	return ok && encoder.SupportsStreamingCollectionEncoding()
}
