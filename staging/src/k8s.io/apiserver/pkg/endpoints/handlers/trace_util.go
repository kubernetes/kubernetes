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

package handlers

import (
	"net/http"

	"go.opentelemetry.io/otel/attribute"
)

func traceFields(req *http.Request) []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.Stringer("accept", &lazyAccept{req: req}),
		attribute.Stringer("audit-id", &lazyAuditID{req: req}),
		attribute.Stringer("client", &lazyClientIP{req: req}),
		attribute.String("protocol", req.Proto),
		attribute.Stringer("resource", &lazyResource{req: req}),
		attribute.Stringer("scope", &lazyScope{req: req}),
		attribute.String("url", req.URL.Path),
		attribute.Stringer("user-agent", &lazyTruncatedUserAgent{req: req}),
		attribute.Stringer("verb", &lazyVerb{req: req}),
	}
}
