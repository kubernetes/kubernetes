/*
Copyright 2017 The Kubernetes Authors.

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

package admission

import (
	"fmt"
	"strconv"
	"time"

	"k8s.io/api/admissionregistration/v1alpha1"

	"github.com/prometheus/client_golang/prometheus"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	namespace = "apiserver"
	subsystem = "admission"
)

var (
	latencyBuckets       = prometheus.ExponentialBuckets(25000, 2.0, 7)
	latencySummaryMaxAge = 5 * time.Hour

	// Metrics provides access to all admission metrics.
	Metrics = newAdmissionMetrics()
)

// NamedHandler requires each admission.Interface be named, primarly for metrics tracking purposes.
type NamedHandler interface {
	Interface() Interface
	Name() string
}

// AdmissionMetrics instruments admission with prometheus metrics.
type AdmissionMetrics struct {
	step            *metricSet
	controller      *metricSet
	externalWebhook *metricSet
}

// newAdmissionMetrics create a new AdmissionMetrics, configured with default metric names.
func newAdmissionMetrics() *AdmissionMetrics {
	// Admission metrics for a step of the admission flow. The entire admission flow is broken down into a series of steps
	// Each step is identified by a distinct type label value.
	step := newMetricSet("step",
		[]string{"type", "operation", "group", "version", "resource", "subresource", "is_system_ns", "rejected"},
		"Admission sub-step %s, broken out for each operation and API resource and step type (validating or mutating).")

	// Built-in admission controller metrics. Each admission controller is identified by name.
	controller := newMetricSet("controller",
		[]string{"name", "type", "operation", "group", "version", "resource", "subresource", "is_system_ns", "rejected"},
		"Admission controller %s, identified by name and broken out for each operation and API resource and type (validating or mutating).")

	// External admission webhook metrics. Each webhook is identified by name.
	externalWebhook := newMetricSet("external_webhook",
		[]string{"name", "type", "operation", "group", "version", "resource", "subresource", "is_system_ns", "rejected"},
		"External admission webhook %s, identified by name and broken out for each operation and API resource and type (validating or mutating).")

	step.mustRegister()
	controller.mustRegister()
	externalWebhook.mustRegister()
	return &AdmissionMetrics{step: step, controller: controller, externalWebhook: externalWebhook}
}

func (m *AdmissionMetrics) reset() {
	m.step.reset()
	m.controller.reset()
	m.externalWebhook.reset()
}

// ObserveAdmissionStep records admission related metrics for a admission step, identified by step type.
func (m *AdmissionMetrics) ObserveAdmissionStep(elapsed time.Duration, rejected bool, attr Attributes, stepType string) {
	gvr := attr.GetResource()
	m.step.observe(elapsed, stepType, string(attr.GetOperation()), gvr.Group, gvr.Version, gvr.Resource, attr.GetSubresource(), isSystemNsLabel(attr), strconv.FormatBool(rejected))
}

// ObserveAdmissionController records admission related metrics for a built-in admission controller, identified by it's plugin handler name.
func (m *AdmissionMetrics) ObserveAdmissionController(elapsed time.Duration, rejected bool, handler NamedHandler, attr Attributes, stepType string) {
	gvr := attr.GetResource()
	m.controller.observe(elapsed, handler.Name(), stepType, string(attr.GetOperation()), gvr.Group, gvr.Version, gvr.Resource, attr.GetSubresource(), isSystemNsLabel(attr), strconv.FormatBool(rejected))
}

// ObserveExternalWebhook records admission related metrics for a external admission webhook.
func (m *AdmissionMetrics) ObserveExternalWebhook(elapsed time.Duration, rejected bool, hook *v1alpha1.ExternalAdmissionHook, attr Attributes) {
	t := "validating" // TODO: pass in type (validating|mutating) once mutating webhook functionality has been implemented
	gvr := attr.GetResource()
	m.externalWebhook.observe(elapsed, hook.Name, t, string(attr.GetOperation()), gvr.Group, gvr.Version, gvr.Resource, attr.GetSubresource(), isSystemNsLabel(attr), strconv.FormatBool(rejected))
}

// isSystemNsLabel returns the value to use for the `is_system_ns` metric label.
func isSystemNsLabel(a Attributes) string {
	return strconv.FormatBool(a.GetNamespace() == metav1.NamespaceSystem)
}

type metricSet struct {
	latencies        *prometheus.HistogramVec
	latenciesSummary *prometheus.SummaryVec
}

func newMetricSet(name string, labels []string, helpTemplate string) *metricSet {
	return &metricSet{
		latencies: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: namespace,
				Subsystem: subsystem,
				Name:      fmt.Sprintf("%s_latencies", name),
				Help:      fmt.Sprintf(helpTemplate, "latency histogram"),
				Buckets:   latencyBuckets,
			},
			labels,
		),
		latenciesSummary: prometheus.NewSummaryVec(
			prometheus.SummaryOpts{
				Namespace: namespace,
				Subsystem: subsystem,
				Name:      fmt.Sprintf("%s_latencies_summary", name),
				Help:      fmt.Sprintf(helpTemplate, "latency summary"),
				MaxAge:    latencySummaryMaxAge,
			},
			labels,
		),
	}
}

// MustRegister registers all the prometheus metrics in the metricSet.
func (m *metricSet) mustRegister() {
	prometheus.MustRegister(m.latencies)
	prometheus.MustRegister(m.latenciesSummary)
}

// Reset resets all the prometheus metrics in the metricSet.
func (m *metricSet) reset() {
	m.latencies.Reset()
	m.latenciesSummary.Reset()
}

// Observe records an observed admission event to all metrics in the metricSet.
func (m *metricSet) observe(elapsed time.Duration, labels ...string) {
	elapsedMicroseconds := float64(elapsed / time.Microsecond)
	m.latencies.WithLabelValues(labels...).Observe(elapsedMicroseconds)
	m.latenciesSummary.WithLabelValues(labels...).Observe(elapsedMicroseconds)
}
