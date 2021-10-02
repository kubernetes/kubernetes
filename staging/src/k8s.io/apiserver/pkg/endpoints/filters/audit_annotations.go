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
	"net/http"

	"k8s.io/apiserver/pkg/audit"
)

// WithAuditAnnotations decorates a http.Handler with a []{key, value} that is merged
// with the audit.Event.Annotations map.  This allows layers that run before WithAudit
// (such as authentication) to assert annotations.
// If sink or audit policy is nil, no decoration takes place.
func WithAuditAnnotations(handler http.Handler, sink audit.Sink, policy audit.PolicyRuleEvaluator) http.Handler {
	// no need to wrap if auditing is disabled
	if sink == nil || policy == nil {
		return handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		req = req.WithContext(audit.WithAuditAnnotations(req.Context()))
		handler.ServeHTTP(w, req)
	})
}
