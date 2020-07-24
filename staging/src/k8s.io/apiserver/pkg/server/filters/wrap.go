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

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/server/httplog"
)

// WithPanicRecovery wraps an http Handler to recover and log panics (except in the special case of http.ErrAbortHandler panics, which suppress logging).
func WithPanicRecovery(handler http.Handler, isTerminating func() bool) http.Handler {
	return withPanicRecovery(handler, func(w http.ResponseWriter, req *http.Request, err interface{}) {
		if err == http.ErrAbortHandler {
			// honor the http.ErrAbortHandler sentinel panic value:
			//   ErrAbortHandler is a sentinel panic value to abort a handler.
			//   While any panic from ServeHTTP aborts the response to the client,
			//   panicking with ErrAbortHandler also suppresses logging of a stack trace to the server's error log.
			return
		}
		http.Error(w, "This request caused apiserver to panic. Look in the logs for details.", http.StatusInternalServerError)
		klog.Errorf("apiserver panic'd on %v %v", req.Method, req.RequestURI)
	}, isTerminating)
}

func withPanicRecovery(handler http.Handler, crashHandler func(http.ResponseWriter, *http.Request, interface{}), isTerminating func() bool) http.Handler {
	handler = httplog.WithLogging(handler, httplog.DefaultStacktracePred, isTerminating)
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer runtime.HandleCrash(func(err interface{}) {
			crashHandler(w, req, err)
		})

		// Dispatch to the internal handler
		handler.ServeHTTP(w, req)
	})
}
