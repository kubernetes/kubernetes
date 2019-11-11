/*
Copyright 2019 The Kubernetes Authors.

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
	"strings"
	"sync"
	"time"

	"k8s.io/apiserver/pkg/authentication/authenticator"
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
	successLabel                 = "success"
	failureLabel                 = "failure"
	errorLabel                   = "error"
	audiencesDoNotIntersectLabel = "non-intersecting-audiences"
	unknownAuthenticator         = "unknown"
)

var (
	authenticatedUserCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "authenticated_user_requests",
			Help:           "Counter of authenticated requests broken out by username.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"username", "method", "authenticator"},
	)

	authenticatedAttemptsCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "authentication_attempts",
			Help:           "Counter of authenticated attempts.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result", "method", "authenticator"},
	)

	authenticationLatency = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Name:           "authentication_duration_seconds",
			Help:           "Authentication duration in seconds broken out by result.",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result", "method", "authenticator"},
	)
)
var once sync.Once

func registerMetrics() {
	once.Do(func() {
		legacyregistry.MustRegister(authenticatedUserCounter)
		legacyregistry.MustRegister(authenticatedAttemptsCounter)
		legacyregistry.MustRegister(authenticationLatency)
	})
}

func recordMetrics(resp *authenticator.Response, ok bool, err error, apiAudiences authenticator.Audiences, authStart time.Time) {
	var (
		label             = successLabel
		authMethod        = authMethodFromResponse(resp)
		authenticatorName = authenticatorNameFromResponse(resp)
		audIntersect      = true
	)

	switch {
	case err != nil:
		klog.Errorf("Unable to authenticate the request due to an error: %v", err)
		label = errorLabel
		authMethod = authMethodFromError(err)
		authenticatorName = authenticatorNameFromError(err)
	case !ok:
		label = failureLabel
	case !audiencesIntersect(apiAudiences, resp.Audiences):
		audIntersect = false
		label = audiencesDoNotIntersectLabel
	}

	if err == nil && ok && audIntersect {
		authenticatedUserCounter.WithLabelValues(compressUsername(resp.User.GetName()), authMethod, authenticatorName).Inc()
	}

	authenticatedAttemptsCounter.WithLabelValues(label, authMethod, authenticatorName).Inc()
	authenticationLatency.WithLabelValues(label, authMethod, authenticatorName).Observe(time.Since(authStart).Seconds())
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

func authMethodFromResponse(response *authenticator.Response) string {
	if response == nil || response.AuthMethod == "" {
		return unknownAuthenticator
	}
	return response.AuthMethod
}

func authMethodFromError(err error) string {
	var authErr *authenticator.Error
	if ok := errors.As(err, &authErr); !ok || authErr.AuthMethod == "" {
		return unknownAuthenticator
	}

	return authErr.AuthMethod
}

func authenticatorNameFromResponse(response *authenticator.Response) string {
	if response == nil || response.AuthenticatorName == "" {
		return unknownAuthenticator
	}
	return response.AuthenticatorName
}

func authenticatorNameFromError(err error) string {
	var authErr *authenticator.Error
	if ok := errors.As(err, &authErr); !ok || authErr.AuthenticatorName == "" {
		return unknownAuthenticator
	}

	return authErr.AuthenticatorName
}
