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

package admission

import (
	"strconv"

	"github.com/prometheus/client_golang/prometheus"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	metricNamespace = "apiserver"
	metricSubsystem = "admission"
)

var (
	handleCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: metricNamespace,
			Subsystem: metricSubsystem,
			Name:      "handle_total",
			Help:      "Counter of all calls to Admit.",
		},
		[]string{"is_system_ns"})
	rejectCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: metricNamespace,
			Subsystem: metricSubsystem,
			Name:      "reject_total",
			Help:      "Counter of all errors returned during admission.",
		},
		[]string{"plugin", "is_system_ns"})
)

func init() {
	prometheus.MustRegister(handleCounter)
	prometheus.MustRegister(rejectCounter)
}

// chainAdmissionHandler is an instance of admission.Interface that performs admission control using a chain of admission handlers
type chainAdmissionHandler []namedAdmissionHandler

type namedAdmissionHandler struct {
	name string
	Interface
}

func (admissionHandler chainAdmissionHandler) Append(name string, handler Interface) chainAdmissionHandler {
	return append(admissionHandler, namedAdmissionHandler{name, handler})
}

// Admit performs an admission control check using a chain of handlers, and returns immediately on first error
func (admissionHandler chainAdmissionHandler) Admit(a Attributes) error {
	handleCounter.WithLabelValues(isSystemNsLabel(a)).Inc()
	for _, handler := range admissionHandler {
		if !handler.Handles(a.GetOperation()) {
			continue
		}
		if mutator, ok := handler.Interface.(MutationInterface); ok {
			err := mutator.Admit(a)
			if err != nil {
				rejectCounter.WithLabelValues(handler.name, isSystemNsLabel(a)).Inc()
				return err
			}
		}
	}
	return nil
}

// Validate performs an admission control check using a chain of handlers, and returns immediately on first error
func (admissionHandler chainAdmissionHandler) Validate(a Attributes) error {
	for _, handler := range admissionHandler {
		if !handler.Handles(a.GetOperation()) {
			continue
		}
		if validator, ok := handler.Interface.(ValidationInterface); ok {
			err := validator.Validate(a)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// Handles will return true if any of the handlers handles the given operation
func (admissionHandler chainAdmissionHandler) Handles(operation Operation) bool {
	for _, handler := range admissionHandler {
		if handler.Handles(operation) {
			return true
		}
	}
	return false
}

// Returns the value to use for the `is_system_ns` metric label.
func isSystemNsLabel(a Attributes) string {
	return strconv.FormatBool(a.GetNamespace() == metav1.NamespaceSystem)
}
