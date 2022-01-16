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
	"fmt"
	"net/http"

	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/httplog"
	"k8s.io/klog/v2"
)

// WithPanicRecovery wraps an http Handler to recover and log panics (except in the special case of http.ErrAbortHandler panics, which suppress logging).
func WithPanicRecovery(handler http.Handler, resolver request.RequestInfoResolver) http.Handler {
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
			}
			// This call can have different handlers, but the default chain rate limits. Call it after the metrics are updated
			// in case the rate limit delays it.  If you outrun the rate for this one timed out requests, something has gone
			// seriously wrong with your server, but generally having a logging signal for timeouts is useful.
			runtime.HandleError(fmt.Errorf("timeout or abort while handling: method=%v URI=%q audit-ID=%q", req.Method, req.RequestURI, request.GetAuditIDTruncated(req)))
			return
		}
		http.Error(w, "This request caused apiserver to panic. Look in the logs for details.", http.StatusInternalServerError)
		klog.ErrorS(nil, "apiserver panic'd", "method", req.Method, "URI", req.RequestURI, "audit-ID", request.GetAuditIDTruncated(req))
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
