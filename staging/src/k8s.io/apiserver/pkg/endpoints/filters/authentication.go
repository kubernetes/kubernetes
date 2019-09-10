/*
Copyright 2014 The Kubernetes Authors.

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
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog"
)

/*
 * By default, all the following metrics are defined as falling under
 * ALPHA stability level https://github.com/kubernetes/enhancements/blob/master/keps/sig-instrumentation/20190404-kubernetes-control-plane-metrics-stability.md#stability-classes)
 *
 * Promoting the stability level of the metric is a responsibility of the component owner, since it
 * involves explicitly acknowledging support for the metric across multiple releases, in accordance with
 * the metric stability policy.
 */
const (
	successLabel = "success"
	failureLabel = "failure"
	errorLabel   = "error"
)

var (
	authenticatedUserCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "authenticated_user_requests",
			Help:           "Counter of authenticated requests broken out by username.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"username"},
	)

	authenticatedAttemptsCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name: "authentication_attempts",
			Help: "Counter of authenticated attempts.",
		},
		[]string{"result"},
	)
)

func init() {
	legacyregistry.MustRegister(authenticatedUserCounter)
	legacyregistry.MustRegister(authenticatedAttemptsCounter)
}

// WithAuthentication creates an http handler that tries to authenticate the given request as a user, and then
// stores any such user found onto the provided context for the request. If authentication fails or returns an error
// the failed handler is used. On success, "Authorization" header is removed from the request and handler
// is invoked to serve the request.
func WithAuthentication(handler http.Handler, auth authenticator.Request, failed http.Handler, apiAuds authenticator.Audiences) http.Handler {
	if auth == nil {
		klog.Warningf("Authentication is disabled")
		return handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if len(apiAuds) > 0 {
			req = req.WithContext(authenticator.WithAudiences(req.Context(), apiAuds))
		}
		resp, ok, err := auth.AuthenticateRequest(req)
		if err != nil || !ok {
			if err != nil {
				klog.Errorf("Unable to authenticate the request due to an error: %v", err)
				authenticatedAttemptsCounter.WithLabelValues(errorLabel).Inc()
			} else if !ok {
				authenticatedAttemptsCounter.WithLabelValues(failureLabel).Inc()
			}

			failed.ServeHTTP(w, req)
			return
		}

		// TODO(mikedanese): verify the response audience matches one of apiAuds if
		// non-empty

		// authorization header is not required anymore in case of a successful authentication.
		req.Header.Del("Authorization")

		req = req.WithContext(genericapirequest.WithUser(req.Context(), resp.User))

		authenticatedUserCounter.WithLabelValues(compressUsername(resp.User.GetName())).Inc()
		authenticatedAttemptsCounter.WithLabelValues(successLabel).Inc()

		handler.ServeHTTP(w, req)
	})
}

func Unauthorized(s runtime.NegotiatedSerializer, supportsBasicAuth bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if supportsBasicAuth {
			w.Header().Set("WWW-Authenticate", `Basic realm="kubernetes-master"`)
		}
		ctx := req.Context()
		requestInfo, found := genericapirequest.RequestInfoFrom(ctx)
		if !found {
			responsewriters.InternalError(w, req, errors.New("no RequestInfo found in the context"))
			return
		}

		gv := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
		responsewriters.ErrorNegotiated(apierrors.NewUnauthorized("Unauthorized"), s, gv, w, req)
	})
}

// compressUsername maps all possible usernames onto a small set of categories
// of usernames. This is done both to limit the cardinality of the
// authorized_user_requests metric, and to avoid pushing actual usernames in the
// metric.
func compressUsername(username string) string {
	switch {
	// Known internal identities.
	case username == "admin" ||
		username == "client" ||
		username == "kube_proxy" ||
		username == "kubelet" ||
		username == "system:serviceaccount:kube-system:default":
		return username
	// Probably an email address.
	case strings.Contains(username, "@"):
		return "email_id"
	// Anything else (custom service accounts, custom external identities, etc.)
	default:
		return "other"
	}
}
