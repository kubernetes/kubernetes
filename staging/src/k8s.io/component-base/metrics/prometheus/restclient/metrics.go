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

package restclient

import (
	"math"
	"net/url"
	"time"

	"k8s.io/client-go/tools/metrics"
	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var (
	// requestLatency is a Prometheus Summary metric type partitioned by
	// "verb" and "url" labels. It is used for the rest client latency metrics.
	requestLatency = k8smetrics.NewHistogramVec(
		&k8smetrics.HistogramOpts{
			Name:    "rest_client_request_duration_seconds",
			Help:    "Request latency in seconds. Broken down by verb and URL.",
			Buckets: k8smetrics.ExponentialBuckets(0.001, 2, 10),
		},
		[]string{"verb", "url"},
	)

	requestResult = k8smetrics.NewCounterVec(
		&k8smetrics.CounterOpts{
			Name: "rest_client_requests_total",
			Help: "Number of HTTP requests, partitioned by status code, method, and host.",
		},
		[]string{"code", "method", "host"},
	)

	execPluginCertTTLAdapter = &expiryToTTLAdapter{}

	execPluginCertTTL = k8smetrics.NewGaugeFunc(
		k8smetrics.GaugeOpts{
			Name: "rest_client_exec_plugin_ttl_seconds",
			Help: "Gauge of the shortest TTL (time-to-live) of the client " +
				"certificate(s) managed by the auth exec plugin. The value " +
				"is in seconds until certificate expiry (negative if " +
				"already expired). If auth exec plugins are unused or manage no " +
				"TLS certificates, the value will be +INF.",
			StabilityLevel: k8smetrics.ALPHA,
		},
		func() float64 {
			if execPluginCertTTLAdapter.e == nil {
				return math.Inf(1)
			}
			return execPluginCertTTLAdapter.e.Sub(time.Now()).Seconds()
		},
	)

	execPluginCertRotation = k8smetrics.NewHistogram(
		&k8smetrics.HistogramOpts{
			Name: "rest_client_exec_plugin_certificate_rotation_age",
			Help: "Histogram of the number of seconds the last auth exec " +
				"plugin client certificate lived before being rotated. " +
				"If auth exec plugin client certificates are unused, " +
				"histogram will contain no data.",
			// There are three sets of ranges these buckets intend to capture:
			//   - 10-60 minutes: captures a rotation cadence which is
			//     happening too quickly.
			//   - 4 hours - 1 month: captures an ideal rotation cadence.
			//   - 3 months - 4 years: captures a rotation cadence which is
			//     is probably too slow or much too slow.
			Buckets: []float64{
				600,       // 10 minutes
				1800,      // 30 minutes
				3600,      // 1  hour
				14400,     // 4  hours
				86400,     // 1  day
				604800,    // 1  week
				2592000,   // 1  month
				7776000,   // 3  months
				15552000,  // 6  months
				31104000,  // 1  year
				124416000, // 4  years
			},
		},
	)
)

func init() {

	legacyregistry.MustRegister(requestLatency)
	legacyregistry.MustRegister(requestResult)
	legacyregistry.RawMustRegister(execPluginCertTTL)
	legacyregistry.MustRegister(execPluginCertRotation)
	metrics.Register(metrics.RegisterOpts{
		ClientCertExpiry:      execPluginCertTTLAdapter,
		ClientCertRotationAge: &rotationAdapter{m: execPluginCertRotation},
		RequestLatency:        &latencyAdapter{m: requestLatency},
		RequestResult:         &resultAdapter{requestResult},
	})
}

type latencyAdapter struct {
	m *k8smetrics.HistogramVec
}

func (l *latencyAdapter) Observe(verb string, u url.URL, latency time.Duration) {
	l.m.WithLabelValues(verb, u.String()).Observe(latency.Seconds())
}

type resultAdapter struct {
	m *k8smetrics.CounterVec
}

func (r *resultAdapter) Increment(code, method, host string) {
	r.m.WithLabelValues(code, method, host).Inc()
}

type expiryToTTLAdapter struct {
	e *time.Time
}

func (e *expiryToTTLAdapter) Set(expiry *time.Time) {
	e.e = expiry
}

type rotationAdapter struct {
	m *k8smetrics.Histogram
}

func (r *rotationAdapter) Observe(d time.Duration) {
	r.m.Observe(d.Seconds())
}
