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

package controllers

import (
	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var (
	// Metrics provides access to all controllerreadiness metrics.
	Metrics = newControllerMetrics()
)

// ControllerMetrics includes all the metrics of the proxy server.
type ControllerMetrics struct {
	controllerInstanceCount *k8smetrics.GaugeVec
}

// newControllerMetrics create a new ControllerMetrics, configured with default metric names.
func newControllerMetrics() *ControllerMetrics {
	controllerInstanceCount := k8smetrics.NewGaugeVec(
		&k8smetrics.GaugeOpts{
			Name:           "managed_controller_instance_count",
			Help:           "Instances of individual controllers currently running",
			StabilityLevel: k8smetrics.ALPHA,
		},
		[]string{"controller_name", "controller_manager"},
	)
	legacyregistry.MustRegister(controllerInstanceCount)
	return &ControllerMetrics{
		controllerInstanceCount: controllerInstanceCount,
	}
}

// controller_name and controller_manager should be updated to validate against bounded lists
// this will allow us to confirm they are populated with "good" values.

// These values use set instead of inc/dec to avoid accidentally double counting
// a controller that starts but fails to properly signal when it crashes.
// ControllerStarted sets the controllerInstanceCount to 1.
func (a *ControllerMetrics) ControllerStarted(controllerName string, controllerManager string) {
	a.controllerInstanceCount.With(k8smetrics.Labels{"controller_name": controllerName, "controller_manager": controllerManager}).Set(float64(1))
}

// ControllerStopped sets the controllerInstanceCount to 0.
func (a *ControllerMetrics) ControllerStopped(controllerName string, controllerManager string) {
	a.controllerInstanceCount.With(k8smetrics.Labels{"controller_name": controllerName, "controller_manager": controllerManager}).Set(float64(0))
}
