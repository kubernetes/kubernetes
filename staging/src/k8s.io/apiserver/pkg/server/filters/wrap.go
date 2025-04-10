/*
Copyright 2016 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/httplog"
	"k8s.io/klog/v2"
)

// WithPanicRecovery wraps a http.Handler to log panic stack traces.
// The (expected) http.ErrAbortHandler panic value is treated specially:
// for long-running requests it is ignored; for short running requests it is
// logged, but without a stack trace.
func WithPanicRecovery(handler http.Handler, resolver request.RequestInfoResolver, longRunning request.LongRunningRequestCheck) http.Handler {
	return withPanicRecovery(handler, func(w http.ResponseWriter, req *http.Request, err interface{}) {
		if err == http.ErrAbortHandler {
			// Honor the http.ErrAbortHandler sentinel panic value
			//
			// If ServeHTTP panics, the server (the caller of ServeHTTP) assumes
			// that the effect of the panic was isolated to the active request.
			// It recovers the panic, logs a stack trace to the server error log,
			// and either closes the network connection or sends an HTTP/2
			// RST_STREAM, depending on the HTTP protocol. To abort a handler so
			// the client sees an interrupted response but the server doesn't log
			// an error, panic with the value ErrAbortHandler.
			//
			// Note that HandleCrash function is actually crashing, after calling the handlers
			if info, err := resolver.NewRequestInfo(req); err != nil {
				metrics.RecordRequestAbort(req, nil)
			} else {
				metrics.RecordRequestAbort(req, info)
				if longRunning != nil && longRunning(req, info) {
					// This was a long-running request such as a watch. Since
					// these get ignored by WithTimeoutForNonLongRunningRequests,
					// the only common cause is that the client closed the
					// connection and the connection was proxied. We don't
					// want to spam the log in that case.
					klog.V(6).InfoS("Ignoring ErrAbortHandler panic for long-running request", "method", req.Method, "URI", req.RequestURI, "auditID", audit.GetAuditIDTruncated(req.Context()))
					return
				}
			}
			// This call can have different handlers, but the default chain rate limits. Call it after the metrics are updated
			// in case the rate limit delays it.  If you outrun the rate for this one timed out requests, something has gone
			// seriously wrong with your server, but generally having a logging signal for timeouts is useful.
			runtime.HandleErrorWithContext(req.Context(), nil, "Timeout or abort while handling", "method", req.Method, "URI", req.RequestURI, "auditID", audit.GetAuditIDTruncated(req.Context()))
			return
		}
		http.Error(w, "This request caused apiserver to panic. Look in the logs for details.", http.StatusInternalServerError)
		klog.ErrorS(nil, "apiserver panic'd", "method", req.Method, "URI", req.RequestURI, "auditID", audit.GetAuditIDTruncated(req.Context()))
	})
}

// WithHTTPLogging enables logging of incoming requests.
func WithHTTPLogging(handler http.Handler) http.Handler {
	return httplog.WithLogging(handler, httplog.DefaultStacktracePred)
}

func withPanicRecovery(handler http.Handler, crashHandler func(http.ResponseWriter, *http.Request, interface{})) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer runtime.HandleCrash(func(err interface{}) {
			crashHandler(w, req, err)
		})

		// Dispatch to the internal handler
		handler.ServeHTTP(w, req)
	})
}
