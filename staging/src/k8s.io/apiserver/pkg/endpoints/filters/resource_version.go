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
	"errors"
	"net/http"
	"strconv"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// WithResourceVersionValidation ensures that the "resource version" query prameter
// if specified is valid.
// If an invalid value of resourceVersion is specified in the request URI, the
// request is rejected with a 400.
func WithResourceVersionValidation(handler http.Handler, negotiatedSerializer runtime.NegotiatedSerializer) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if !hasValidResourceVersion(req) {
			ctx := req.Context()
			requestInfo, found := request.RequestInfoFrom(ctx)
			if !found {
				responsewriters.InternalError(w, req, errors.New("no RequestInfo found in the context"))
				return
			}

			statusErr := apierrors.NewBadRequest("resourceVersion specified in the request URI is invalid")

			gv := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
			responsewriters.ErrorNegotiated(statusErr, negotiatedSerializer, gv, w, req)
			return
		}

		handler.ServeHTTP(w, req)
	})
}

func hasValidResourceVersion(r *http.Request) bool {
	rv := r.URL.Query().Get("resourceVersion")
	if rv == "" {
		// resourceVersion can be empty.
		return true
	}

	if _, err := strconv.ParseInt(rv, 0, 64); err != nil {
		return false
	}

	return true
}
