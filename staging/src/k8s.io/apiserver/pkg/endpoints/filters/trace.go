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
	"context"
	"net/http"
	"time"

	utiltrace "k8s.io/utils/trace"

	"k8s.io/apiserver/pkg/endpoints/internal"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

// WithTrace decorates a http.Handler with tracing for all the non-long running
// requests coming to the server.
func WithTrace(handler http.Handler, longRunningCheck genericapirequest.LongRunningRequestCheck) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ri, ok := genericapirequest.RequestInfoFrom(req.Context())
		isLongRunning := false
		if longRunningCheck != nil {
			if ok && longRunningCheck(req, ri) {
				isLongRunning = true
			}
		}

		if isLongRunning {
			handler.ServeHTTP(w, req)
			return
		}
		// Set trace as root trace in context for nested tracing
		ctx, trace := genericapirequest.WithTrace(req.Context(), "HTTP Request",
			utiltrace.Field{Key: "method", Value: req.Method},
			utiltrace.Field{Key: "url", Value: req.URL.Path},
			utiltrace.Field{Key: "verb", Value: ri.Verb},
			utiltrace.Field{Key: "name", Value: ri.Name},
			utiltrace.Field{Key: "resource", Value: ri.Resource},
			utiltrace.Field{Key: "subresource", Value: ri.Subresource},
			utiltrace.Field{Key: "namespace", Value: ri.Namespace},
			utiltrace.Field{Key: "api-group", Value: ri.APIGroup},
			utiltrace.Field{Key: "api-version", Value: ri.APIVersion},
			utiltrace.Field{Key: "user-agent", Value: &internal.LazyTruncatedUserAgent{req}},
			utiltrace.Field{Key: "client", Value: &internal.LazyClientIP{req}})
		req = req.Clone(ctx)
		defer func(start time.Time) {
			if time.Since(start) >= 30*time.Second {
				// To mitigate https://github.com/kubernetes/kubernetes/issues/93042
				// don't call LogIfLong here unless threshold is exceeded. This avoid printing
				// this trace unconditionally when verbosity is set above 4, which is the case for
				// e2e tests. LogIfLong called here instead of Log so that individual steps are
				// only logged if step threshold is exceeded.
				trace.LogIfLong(30 * time.Second)
			}
		}(time.Now())
		handler.ServeHTTP(w, req)
	})
}

func traceFilterStep(ctx context.Context, msg string, field ...utiltrace.Field) {
	if trace, ok := genericapirequest.TraceFrom(ctx); ok {
		trace.Step(msg, field...)
	}
}
