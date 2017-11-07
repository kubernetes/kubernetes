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
	latencyBuckets       = prometheus.ExponentialBuckets(10000, 2.0, 8)
	latencySummaryMaxAge = 5 * time.Hour

	Metrics = newAdmissionMetrics()
)

type AdmissionMetrics struct {
	step            *metricSet
	controller      *metricSet
	externalWebhook *metricSet
}

func newAdmissionMetrics() *AdmissionMetrics {
	// Admission metrics for a step of the admission flow. The entire admission flow is broken down into a series of steps
	// Each step is identified by a distinct type label value.
	step := newMetricSet("step_",
		[]string{"operation", "group", "version", "resource", "subresource", "type", "is_system_ns"},
		"Admission sub-step %s, broken out for each operation and API resource and step type (validating or mutating).")

	// Built-in admission controller metrics. Each admission controller is identified by name.
	controller := newMetricSet("controller_",
		[]string{"name", "type", "operation", "group", "version", "resource", "subresource", "is_system_ns"},
		"Admission controller %s, identified by name and broken out for each operation and API resource and type (validating or mutating).")

	// External admission webhook metrics. Each webhook is identified by name.
	externalWebhook := newMetricSet("external_webhook_",
		[]string{"name", "type", "operation", "group", "version", "resource", "subresource", "is_system_ns"},
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

// namedHandler requires each admission.Interface be named, primarly for metrics tracking purposes.
type NamedHandler interface {
	Interface
	GetName() string
}

// ObserveAdmissionStep records admission related metrics for a admission step, identified by step type.
func (m *AdmissionMetrics) ObserveAdmissionStep(elapsed time.Duration, rejected bool, attr Attributes, stepType string) {
	gvr := attr.GetResource()
	m.step.observe(elapsed, rejected, string(attr.GetOperation()), gvr.Group, gvr.Version, gvr.Resource, attr.GetSubresource(), stepType, isSystemNsLabel(attr))
}

// ObserveAdmissionController records admission related metrics for a built-in admission controller, identified by it's plugin handler name.
func (m *AdmissionMetrics) ObserveAdmissionController(elapsed time.Duration, rejected bool, handler NamedHandler, attr Attributes) {
	t := typeToLabel(handler)
	gvr := attr.GetResource()
	m.controller.observe(elapsed, rejected, handler.GetName(), t, string(attr.GetOperation()), gvr.Group, gvr.Version, gvr.Resource, attr.GetSubresource(), isSystemNsLabel(attr))
}

// ObserveExternalWebhook records admission related metrics for a external admission webhook.
func (m *AdmissionMetrics) ObserveExternalWebhook(elapsed time.Duration, rejected bool, hook *v1alpha1.ExternalAdmissionHook, attr Attributes) {
	t := "validating" // TODO: pass in type (validating|mutating) once mutating webhook functionality has been implemented
	gvr := attr.GetResource()
	m.externalWebhook.observe(elapsed, rejected, hook.Name, t, string(attr.GetOperation()), gvr.Group, gvr.Version, gvr.Resource, attr.GetSubresource(), isSystemNsLabel(attr))
}

// isSystemNsLabel returns the value to use for the `is_system_ns` metric label.
func isSystemNsLabel(a Attributes) string {
	return strconv.FormatBool(a.GetNamespace() == metav1.NamespaceSystem)
}

func typeToLabel(i Interface) string {
	switch i.(type) {
	case MutationInterface:
		return "mutating"
	case ValidationInterface:
		return "validating"
	default:
		return "UNRECOGNIZED_ADMISSION_TYPE"
	}
}

type metricSet struct {
	total            *prometheus.CounterVec
	rejectedTotal    *prometheus.CounterVec
	latencies        *prometheus.HistogramVec
	latenciesSummary *prometheus.SummaryVec
}

func (m *metricSet) mustRegister() {
	prometheus.MustRegister(m.total)
	prometheus.MustRegister(m.rejectedTotal)
	prometheus.MustRegister(m.latencies)
	prometheus.MustRegister(m.latenciesSummary)
}

func (m *metricSet) reset() {
	m.total.Reset()
	m.rejectedTotal.Reset()
	m.latencies.Reset()
	m.latenciesSummary.Reset()
}

func (m *metricSet) observe(elapsed time.Duration, rejected bool, labels ...string) {
	elapsedMicroseconds := float64(elapsed / time.Microsecond)
	m.total.WithLabelValues(labels...).Inc()
	if rejected {
		m.rejectedTotal.WithLabelValues(labels...).Inc()
	}
	m.latencies.WithLabelValues(labels...).Observe(elapsedMicroseconds)
	m.latenciesSummary.WithLabelValues(labels...).Observe(elapsedMicroseconds)
}

func newMetricSet(name string, labels []string, helpTemplate string) *metricSet {
	return &metricSet{
		total: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Subsystem: subsystem,
				Name:      fmt.Sprintf("%stotal", name),
				Help:      fmt.Sprintf(helpTemplate, "count"),
			},
			labels,
		),
		rejectedTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Subsystem: subsystem,
				Name:      fmt.Sprintf("%srejected_total", name),
				Help:      fmt.Sprintf(helpTemplate, "rejected count"),
			},
			labels,
		),
		latencies: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: namespace,
				Subsystem: subsystem,
				Name:      fmt.Sprintf("%slatencies", name),
				Help:      fmt.Sprintf(helpTemplate, "latency histogram"),
				Buckets:   latencyBuckets,
			},
			labels,
		),
		latenciesSummary: prometheus.NewSummaryVec(
			prometheus.SummaryOpts{
				Namespace: namespace,
				Subsystem: subsystem,
				Name:      fmt.Sprintf("%slatencies_summary", name),
				Help:      fmt.Sprintf(helpTemplate, "latency summary"),
				MaxAge:    latencySummaryMaxAge,
			},
			labels,
		),
	}
}
