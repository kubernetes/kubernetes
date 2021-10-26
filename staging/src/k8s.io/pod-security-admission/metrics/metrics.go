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

package metrics

import (
	"strconv"
	"strings"
	"sync"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/pod-security-admission/api"
)

const (
	ModeAudit     = "audit"
	ModeEnforce   = "enforce"
	ModeWarn      = "warn"
	DecisionAllow = "allow" // Policy evaluated, request allowed
	DecisionDeny  = "deny"  // Policy evaluated, request denied
)

type Decision string
type Mode string

type Recorder interface {
	RecordEvaluation(Decision, api.LevelVersion, Mode, api.Attributes)
	RecordExemption(api.Attributes)
	RecordError(fatal bool, attrs api.Attributes)
}

var defaultRecorder = NewPrometheusRecorder(api.GetAPIVersion())

func DefaultRecorder() Recorder {
	return defaultRecorder
}

// MustRegister registers the global DefaultMetrics against the legacy registry.
func LegacyMustRegister() {
	defaultRecorder.MustRegister(legacyregistry.MustRegister)
}

type PrometheusRecorder struct {
	apiVersion api.Version

	evaluationsCounter *metrics.CounterVec
	exemptionsCounter  *metrics.CounterVec
	errorsCounter      *metrics.CounterVec

	registerOnce sync.Once
}

var _ Recorder = &PrometheusRecorder{}

func NewPrometheusRecorder(version api.Version) *PrometheusRecorder {
	evaluationsCounter := metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "pod_security_evaluations_total",
			Help:           "Number of policy evaluations that occurred, not counting ignored or exempt requests.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"decision", "policy_level", "policy_version", "mode", "request_operation", "resource", "subresource"},
	)
	exemptionsCounter := metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "pod_security_exemptions_total",
			Help:           "Number of exempt requests, not counting ignored or out of scope requests.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"request_operation", "resource", "subresource"},
	)
	errorsCounter := metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "pod_security_errors_total",
			Help:           "Number of errors prevent normal evaluation. Non-fatal errors are evaluated against a default policy.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"fatal", "request_operation", "resource", "subresource"},
	)

	return &PrometheusRecorder{
		apiVersion:         version,
		evaluationsCounter: evaluationsCounter,
		exemptionsCounter:  exemptionsCounter,
		errorsCounter:      errorsCounter,
	}
}

func (r *PrometheusRecorder) MustRegister(registerFunc func(...metrics.Registerable)) {
	r.registerOnce.Do(func() {
		registerFunc(r.evaluationsCounter)
		registerFunc(r.exemptionsCounter)
		registerFunc(r.errorsCounter)
	})
}

func (r *PrometheusRecorder) Reset() {
	r.evaluationsCounter.Reset()
	r.exemptionsCounter.Reset()
	r.errorsCounter.Reset()
}

func (r *PrometheusRecorder) RecordEvaluation(decision Decision, policy api.LevelVersion, evalMode Mode, attrs api.Attributes) {
	dec := string(decision)
	operation := operationLabel(attrs.GetOperation())
	resource := resourceLabel(attrs.GetResource())
	subresource := attrs.GetSubresource()

	var version string
	if policy.Version.Latest() {
		version = "latest"
	} else {
		if !r.apiVersion.Older(policy.Version) {
			version = policy.Version.String()
		} else {
			version = "future"
		}
	}

	r.evaluationsCounter.WithLabelValues(dec, string(policy.Level),
		version, string(evalMode), operation, resource, subresource).Inc()
}

func (r *PrometheusRecorder) RecordExemption(attrs api.Attributes) {
	operation := operationLabel(attrs.GetOperation())
	resource := resourceLabel(attrs.GetResource())
	subresource := attrs.GetSubresource()
	r.exemptionsCounter.WithLabelValues(operation, resource, subresource).Inc()
}

func (r *PrometheusRecorder) RecordError(fatal bool, attrs api.Attributes) {
	operation := operationLabel(attrs.GetOperation())
	resource := resourceLabel(attrs.GetResource())
	subresource := attrs.GetSubresource()
	r.errorsCounter.WithLabelValues(strconv.FormatBool(fatal), operation, resource, subresource).Inc()
}

func resourceLabel(resource schema.GroupVersionResource) string {
	switch resource.GroupResource() {
	case corev1.Resource("pods"):
		return "pod"
	case corev1.Resource("namespace"):
		return "namespace"
	default:
		// Assume any other resource is a valid input to pod-security, and therefore a controller.
		return "controller"
	}
}

func operationLabel(op admissionv1.Operation) string {
	switch op {
	case admissionv1.Create:
		return "create"
	case admissionv1.Update:
		return "update"
	default:
		// This is a slower operation, but never used in the default implementation.
		return strings.ToLower(string(op))
	}
}
