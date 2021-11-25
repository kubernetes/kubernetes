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

	"k8s.io/apimachinery/pkg/types"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/endpoints/request"

	"github.com/google/uuid"
)

// WithAuditID attaches the Audit-ID associated with a request to the context.
//
// a. If the caller does not specify a value for Audit-ID in the request header, we generate a new audit ID
// b. We echo the Audit-ID value to the caller via the response Header 'Audit-ID'.
func WithAuditID(handler http.Handler) http.Handler {
	return withAuditID(handler, func() string {
		return uuid.New().String()
	})
}

func withAuditID(handler http.Handler, newAuditIDFunc func() string) http.Handler {
	if newAuditIDFunc == nil {
		return handler
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()

		auditID := r.Header.Get(auditinternal.HeaderAuditID)
		if len(auditID) == 0 {
			auditID = newAuditIDFunc()
		}

		// Note: we save the user specified value of the Audit-ID header as is, no truncation is performed.
		r = r.WithContext(request.WithAuditID(ctx, types.UID(auditID)))

		// We echo the Audit-ID in to the response header.
		// It's not guaranteed Audit-ID http header is sent for all requests.
		// For example, when user run "kubectl exec", apiserver uses a proxy handler
		// to deal with the request, users can only get http headers returned by kubelet node.
		//
		// This filter will also be used by other aggregated api server(s). For an aggregated API
		// we don't want to see the same audit ID appearing more than once.
		if value := w.Header().Get(auditinternal.HeaderAuditID); len(value) == 0 {
			w.Header().Set(auditinternal.HeaderAuditID, auditID)
		}

		handler.ServeHTTP(w, r)
	})
}
