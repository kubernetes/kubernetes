/*
Copyright 2023 The Kubernetes Authors.

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
	"errors"
	"net/http"

	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog/v2"
)

func WithWatchTerminationDuringShutdown(handler http.Handler, termination apirequest.ServerShutdownSignal, wg RequestWaitGroup) http.Handler {
	if termination == nil || wg == nil {
		klog.Warningf("watch termination during shutdown not attached to the handler chain")
		return handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		requestInfo, ok := apirequest.RequestInfoFrom(ctx)
		if !ok {
			// if this happens, the handler chain isn't setup correctly because there is no request info
			responsewriters.InternalError(w, req, errors.New("no RequestInfo found in the context"))
			return
		}
		if !watchVerbs.Has(requestInfo.Verb) {
			handler.ServeHTTP(w, req)
			return
		}

		if err := wg.Add(1); err != nil {
			// When apiserver is shutting down, signal clients to retry
			// There is a good chance the client hit a different server, so a tight retry is good for client responsiveness.
			waitGroupWriteRetryAfterToResponse(w)
			return
		}

		// attach ServerShutdownSignal to the watch request so that the
		// watch handler loop can return as soon as the server signals
		// that it is shutting down.
		ctx = apirequest.WithServerShutdownSignal(req.Context(), termination)
		req = req.WithContext(ctx)

		defer wg.Done()
		handler.ServeHTTP(w, req)
	})
}
