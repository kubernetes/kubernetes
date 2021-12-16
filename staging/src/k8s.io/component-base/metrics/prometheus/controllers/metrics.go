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
	"sync"

	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var (
	once sync.Once
)

// ControllerMetrics includes all the metrics of the proxy server.
type ControllerMetrics struct {
	controllerInstanceCount *k8smetrics.GaugeVec
}

// NewControllerMetrics create a new ControllerMetrics, configured with default metric names.
func NewControllerMetrics() *ControllerMetrics {
	controllerInstanceCount := k8smetrics.NewGaugeVec(
		&k8smetrics.GaugeOpts{
			Name:           "managed_controller_instances",
			Help:           "Indicates where instances of a controller are currently running",
			StabilityLevel: k8smetrics.ALPHA,
		},
		[]string{"name", "manager"},
	)
	controllerMetrics := &ControllerMetrics{
		controllerInstanceCount: controllerInstanceCount,
	}
	once.Do(func() {
		controllerMetrics.Register()
	})
	return controllerMetrics
}

func (a *ControllerMetrics) Register() {
	legacyregistry.MustRegister(a.controllerInstanceCount)
}

// ControllerStarted sets the controllerInstanceCount to 1.
// These values use set instead of inc/dec to avoid accidentally double counting
// a controller that starts but fails to properly signal when it crashes.
func (a *ControllerMetrics) ControllerStarted(name string, manager string) {
	a.controllerInstanceCount.With(k8smetrics.Labels{"name": name, "manager": manager}).Set(float64(1))
}

// ControllerStopped sets the controllerInstanceCount to 0.
func (a *ControllerMetrics) ControllerStopped(name string, manager string) {
	a.controllerInstanceCount.With(k8smetrics.Labels{"name": name, "manager": manager}).Set(float64(0))
}
