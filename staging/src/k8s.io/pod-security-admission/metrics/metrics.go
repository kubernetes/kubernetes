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

	"github.com/blang/semver/v4"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/component-base/metrics"
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

type PrometheusRecorder struct {
	apiVersion api.Version

	evaluationsCounter *evaluationsCounter
	exemptionsCounter  *exemptionsCounter
	errorsCounter      *metrics.CounterVec
}

var _ Recorder = &PrometheusRecorder{}

func NewPrometheusRecorder(version api.Version) *PrometheusRecorder {
	errorsCounter := metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "pod_security_errors_total",
			Help:           "Number of errors preventing normal evaluation. Non-fatal errors may result in the latest restricted profile being used for evaluation.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"fatal", "request_operation", "resource", "subresource"},
	)

	return &PrometheusRecorder{
		apiVersion:         version,
		evaluationsCounter: newEvaluationsCounter(),
		exemptionsCounter:  newExemptionsCounter(),
		errorsCounter:      errorsCounter,
	}
}

func (r *PrometheusRecorder) MustRegister(registerFunc func(...metrics.Registerable)) {
	registerFunc(r.evaluationsCounter)
	registerFunc(r.exemptionsCounter)
	registerFunc(r.errorsCounter)
}

func (r *PrometheusRecorder) Reset() {
	r.evaluationsCounter.Reset()
	r.exemptionsCounter.Reset()
	r.errorsCounter.Reset()
}

func (r *PrometheusRecorder) RecordEvaluation(decision Decision, policy api.LevelVersion, evalMode Mode, attrs api.Attributes) {
	var version string
	if policy.Version.Latest() || policy.Level == api.LevelPrivileged { // Privileged is always effectively latest.
		version = "latest"
	} else {
		if !r.apiVersion.Older(policy.Version) {
			version = policy.Version.String()
		} else {
			version = "future"
		}
	}

	// prevent cardinality explosion by only recording the platform namespaces
	namespace := attrs.GetNamespace()
	if !(namespace == "openshift" ||
		strings.HasPrefix(namespace, "openshift-") ||
		strings.HasPrefix(namespace, "kube-") ||
		namespace == "default") {
		// remove non-OpenShift platform namespace names to prevent cardinality explosion
		namespace = ""
	}

	el := evaluationsLabels{
		decision:     string(decision),
		level:        string(policy.Level),
		version:      version,
		mode:         string(evalMode),
		operation:    operationLabel(attrs.GetOperation()),
		resource:     resourceLabel(attrs.GetResource()),
		subresource:  attrs.GetSubresource(),
		ocpNamespace: namespace,
	}

	r.evaluationsCounter.CachedInc(el)
}

func (r *PrometheusRecorder) RecordExemption(attrs api.Attributes) {
	r.exemptionsCounter.CachedInc(exemptionsLabels{
		operation:   operationLabel(attrs.GetOperation()),
		resource:    resourceLabel(attrs.GetResource()),
		subresource: attrs.GetSubresource(),
	})
}

func (r *PrometheusRecorder) RecordError(fatal bool, attrs api.Attributes) {
	r.errorsCounter.WithLabelValues(
		strconv.FormatBool(fatal),
		operationLabel(attrs.GetOperation()),
		resourceLabel(attrs.GetResource()),
		attrs.GetSubresource(),
	).Inc()
}

var (
	podResource       = corev1.Resource("pods")
	namespaceResource = corev1.Resource("namespaces")
)

func resourceLabel(resource schema.GroupVersionResource) string {
	switch resource.GroupResource() {
	case podResource:
		return "pod"
	case namespaceResource:
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

type evaluationsLabels struct {
	decision     string
	level        string
	version      string
	mode         string
	operation    string
	resource     string
	subresource  string
	ocpNamespace string
}

func (l *evaluationsLabels) labels() []string {
	return []string{l.decision, l.level, l.version, l.mode, l.operation, l.resource, l.subresource, l.ocpNamespace}
}

type exemptionsLabels struct {
	operation   string
	resource    string
	subresource string
}

func (l *exemptionsLabels) labels() []string {
	return []string{l.operation, l.resource, l.subresource}
}

type evaluationsCounter struct {
	*metrics.CounterVec

	cache     map[evaluationsLabels]metrics.CounterMetric
	cacheLock sync.RWMutex
}

func newEvaluationsCounter() *evaluationsCounter {
	return &evaluationsCounter{
		CounterVec: metrics.NewCounterVec(
			&metrics.CounterOpts{
				Name:           "pod_security_evaluations_total",
				Help:           "Number of policy evaluations that occurred, not counting ignored or exempt requests.",
				StabilityLevel: metrics.ALPHA,
			},
			[]string{"decision", "policy_level", "policy_version", "mode", "request_operation", "resource", "subresource", "ocp_namespace"},
		),
		cache: make(map[evaluationsLabels]metrics.CounterMetric),
	}
}

func (c *evaluationsCounter) CachedInc(l evaluationsLabels) {
	c.cacheLock.RLock()
	defer c.cacheLock.RUnlock()

	if cachedCounter, ok := c.cache[l]; ok {
		cachedCounter.Inc()
	} else {
		c.CounterVec.WithLabelValues(l.labels()...).Inc()
	}
}

func (c *evaluationsCounter) Create(version *semver.Version) bool {
	c.cacheLock.Lock()
	defer c.cacheLock.Unlock()
	if c.CounterVec.Create(version) {
		c.populateCache()
		return true
	} else {
		return false
	}
}

func (c *evaluationsCounter) Reset() {
	c.cacheLock.Lock()
	defer c.cacheLock.Unlock()
	c.CounterVec.Reset()
	c.populateCache()
}

func (c *evaluationsCounter) populateCache() {
	labelsToCache := []evaluationsLabels{
		{decision: "allow", level: "privileged", version: "latest", mode: "enforce", operation: "create", resource: "pod", subresource: "", ocpNamespace: ""},
		{decision: "allow", level: "privileged", version: "latest", mode: "enforce", operation: "update", resource: "pod", subresource: "", ocpNamespace: ""},
	}
	for _, l := range labelsToCache {
		c.cache[l] = c.CounterVec.WithLabelValues(l.labels()...)
	}
}

type exemptionsCounter struct {
	*metrics.CounterVec

	cache     map[exemptionsLabels]metrics.CounterMetric
	cacheLock sync.RWMutex
}

func newExemptionsCounter() *exemptionsCounter {
	return &exemptionsCounter{
		CounterVec: metrics.NewCounterVec(
			&metrics.CounterOpts{
				Name:           "pod_security_exemptions_total",
				Help:           "Number of exempt requests, not counting ignored or out of scope requests.",
				StabilityLevel: metrics.ALPHA,
			},
			[]string{"request_operation", "resource", "subresource"},
		),
		cache: make(map[exemptionsLabels]metrics.CounterMetric),
	}
}

func (c *exemptionsCounter) CachedInc(l exemptionsLabels) {
	c.cacheLock.RLock()
	defer c.cacheLock.RUnlock()

	if cachedCounter, ok := c.cache[l]; ok {
		cachedCounter.Inc()
	} else {
		c.CounterVec.WithLabelValues(l.labels()...).Inc()
	}
}

func (c *exemptionsCounter) Create(version *semver.Version) bool {
	c.cacheLock.Lock()
	defer c.cacheLock.Unlock()
	if c.CounterVec.Create(version) {
		c.populateCache()
		return true
	} else {
		return false
	}
}

func (c *exemptionsCounter) Reset() {
	c.cacheLock.Lock()
	defer c.cacheLock.Unlock()
	c.CounterVec.Reset()
	c.populateCache()
}

func (c *exemptionsCounter) populateCache() {
	labelsToCache := []exemptionsLabels{
		{operation: "create", resource: "pod", subresource: ""},
		{operation: "update", resource: "pod", subresource: ""},
		{operation: "create", resource: "controller", subresource: ""},
		{operation: "update", resource: "controller", subresource: ""},
	}
	for _, l := range labelsToCache {
		c.cache[l] = c.CounterVec.WithLabelValues(l.labels()...)
	}
}
