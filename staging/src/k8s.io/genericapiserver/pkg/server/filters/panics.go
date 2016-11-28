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
	"runtime/debug"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apiserver/request"
	"k8s.io/kubernetes/pkg/httplog"
	"k8s.io/kubernetes/pkg/util/runtime"
)

// WithPanicRecovery wraps an http Handler to recover and log panics.
func WithPanicRecovery(handler http.Handler, requestContextMapper api.RequestContextMapper) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer runtime.HandleCrash(func(err interface{}) {
			http.Error(w, "This request caused apisever to panic. Look in log for details.", http.StatusInternalServerError)
			glog.Errorf("APIServer panic'd on %v %v: %v\n%s\n", req.Method, req.RequestURI, err, debug.Stack())
		})

		logger := httplog.NewLogged(req, &w)

		var requestInfo *request.RequestInfo
		ctx, ok := requestContextMapper.Get(req)
		if !ok {
			glog.Errorf("no context found for request, handler chain must be wrong")
		} else {
			requestInfo, ok = request.RequestInfoFrom(ctx)
			if !ok {
				glog.Errorf("no RequestInfo found in context, handler chain must be wrong")
			}
		}

		if !ok || requestInfo.Verb != "proxy" {
			logger.StacktraceWhen(
				httplog.StatusIsNot(
					http.StatusOK,
					http.StatusCreated,
					http.StatusAccepted,
					http.StatusBadRequest,
					http.StatusMovedPermanently,
					http.StatusTemporaryRedirect,
					http.StatusConflict,
					http.StatusNotFound,
					http.StatusUnauthorized,
					http.StatusForbidden,
					http.StatusNotModified,
					apierrors.StatusUnprocessableEntity,
					http.StatusSwitchingProtocols,
				),
			)
		}
		defer logger.Log()

		// Dispatch to the internal handler
		handler.ServeHTTP(w, req)
	})
}
