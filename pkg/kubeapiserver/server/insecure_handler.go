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

package server

import (
	"net/http"

	"k8s.io/apiserver/pkg/authentication/user"
	genericapifilters "k8s.io/apiserver/pkg/endpoints/filters"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server"
	genericfilters "k8s.io/apiserver/pkg/server/filters"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

// DeprecatedInsecureServingInfo is required to serve http.  HTTP does NOT include authentication or authorization.
// You shouldn't be using this.  It makes sig-auth sad.
// DeprecatedInsecureServingInfo *ServingInfo

func BuildInsecureHandlerChain(apiHandler http.Handler, c *server.Config) http.Handler {
	handler := apiHandler
	if utilfeature.DefaultFeatureGate.Enabled(features.AdvancedAuditing) {
		handler = genericapifilters.WithAudit(handler, c.AuditBackend, c.AuditPolicyChecker, c.LongRunningFunc)
	} else {
		handler = genericapifilters.WithLegacyAudit(handler, c.LegacyAuditWriter)
	}
	handler = genericapifilters.WithAuthentication(handler, insecureSuperuser{}, nil)
	handler = genericfilters.WithCORS(handler, c.CorsAllowedOriginList, nil, nil, nil, "true")
	handler = genericfilters.WithTimeoutForNonLongRunningRequests(handler, c.LongRunningFunc, c.RequestTimeout)
	handler = genericfilters.WithMaxInFlightLimit(handler, c.MaxRequestsInFlight, c.MaxMutatingRequestsInFlight, c.LongRunningFunc)
	handler = genericfilters.WithWaitGroup(handler, c.LongRunningFunc, c.HandlerChainWaitGroup)
	handler = genericapifilters.WithRequestInfo(handler, server.NewRequestInfoResolver(c))
	handler = genericfilters.WithPanicRecovery(handler)

	return handler
}

// insecureSuperuser implements authenticator.Request to always return a superuser.
// This is functionally equivalent to skipping authentication and authorization,
// but allows apiserver code to stop special-casing a nil user to skip authorization checks.
type insecureSuperuser struct{}

func (insecureSuperuser) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	return &user.DefaultInfo{
		Name:   "system:unsecured",
		Groups: []string{user.SystemPrivilegedGroup, user.AllAuthenticated},
	}, true, nil
}
