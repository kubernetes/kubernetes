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
	"context"
	"strings"
	"time"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

/*
 * By default, all the following metrics are defined as falling under
 * ALPHA stability level https://github.com/kubernetes/enhancements/blob/master/keps/sig-instrumentation/1209-metrics-stability/kubernetes-control-plane-metrics-stability.md#stability-classes)
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
			Name:           "authentication_attempts",
			Help:           "Counter of authenticated attempts.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result"},
	)

	authenticationLatency = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Name:           "authentication_duration_seconds",
			Help:           "Authentication duration in seconds broken out by result.",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result"},
	)
)

func init() {
	legacyregistry.MustRegister(authenticatedUserCounter)
	legacyregistry.MustRegister(authenticatedAttemptsCounter)
	legacyregistry.MustRegister(authenticationLatency)
}

func recordAuthMetrics(ctx context.Context, resp *authenticator.Response, ok bool, err error, apiAudiences authenticator.Audiences, authStart time.Time, authFinish time.Time) {
	var resultLabel string

	switch {
	case err != nil || (resp != nil && !audiencesAreAcceptable(apiAudiences, resp.Audiences)):
		resultLabel = errorLabel
	case !ok:
		resultLabel = failureLabel
	default:
		resultLabel = successLabel
		authenticatedUserCounter.WithContext(ctx).WithLabelValues(compressUsername(resp.User.GetName())).Inc()
	}

	authenticatedAttemptsCounter.WithContext(ctx).WithLabelValues(resultLabel).Inc()
	authenticationLatency.WithContext(ctx).WithLabelValues(resultLabel).Observe(authFinish.Sub(authStart).Seconds())
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
