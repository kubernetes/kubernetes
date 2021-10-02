/*
Copyright 2018 The Kubernetes Authors.

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

package apiserver

import (
	"sync"

	"k8s.io/component-base/metrics"
)

/*
 * By default, all the following metrics are defined as falling under
 * ALPHA stability level https://github.com/kubernetes/enhancements/blob/master/keps/sig-instrumentation/1209-metrics-stability/kubernetes-control-plane-metrics-stability.md#stability-classes)
 *
 * Promoting the stability level of the metric is a responsibility of the component owner, since it
 * involves explicitly acknowledging support for the metric across multiple releases, in accordance with
 * the metric stability policy.
 */
var (
	unavailableGaugeDesc = metrics.NewDesc(
		"aggregator_unavailable_apiservice",
		"Gauge of APIServices which are marked as unavailable broken down by APIService name.",
		[]string{"name"},
		nil,
		metrics.ALPHA,
		"",
	)
)

type availabilityMetrics struct {
	unavailableCounter *metrics.CounterVec

	*availabilityCollector
}

func newAvailabilityMetrics() *availabilityMetrics {
	return &availabilityMetrics{
		unavailableCounter: metrics.NewCounterVec(
			&metrics.CounterOpts{
				Name:           "aggregator_unavailable_apiservice_total",
				Help:           "Counter of APIServices which are marked as unavailable broken down by APIService name and reason.",
				StabilityLevel: metrics.ALPHA,
			},
			[]string{"name", "reason"},
		),
		availabilityCollector: newAvailabilityCollector(),
	}
}

// Register registers apiservice availability metrics.
func (m *availabilityMetrics) Register(
	registrationFunc func(metrics.Registerable) error,
	customRegistrationFunc func(metrics.StableCollector) error,
) error {
	err := registrationFunc(m.unavailableCounter)
	if err != nil {
		return err
	}

	err = customRegistrationFunc(m.availabilityCollector)
	if err != nil {
		return err
	}

	return nil
}

// UnavailableCounter returns a counter to track apiservices marked as unavailable.
func (m *availabilityMetrics) UnavailableCounter(apiServiceName, reason string) metrics.CounterMetric {
	return m.unavailableCounter.WithLabelValues(apiServiceName, reason)
}

type availabilityCollector struct {
	metrics.BaseStableCollector

	mtx            sync.RWMutex
	availabilities map[string]bool
}

// Check if apiServiceStatusCollector implements necessary interface.
var _ metrics.StableCollector = &availabilityCollector{}

func newAvailabilityCollector() *availabilityCollector {
	return &availabilityCollector{
		availabilities: make(map[string]bool),
	}
}

// DescribeWithStability implements the metrics.StableCollector interface.
func (c *availabilityCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- unavailableGaugeDesc
}

// CollectWithStability implements the metrics.StableCollector interface.
func (c *availabilityCollector) CollectWithStability(ch chan<- metrics.Metric) {
	c.mtx.RLock()
	defer c.mtx.RUnlock()

	for apiServiceName, isAvailable := range c.availabilities {
		gaugeValue := 1.0
		if isAvailable {
			gaugeValue = 0.0
		}
		ch <- metrics.NewLazyConstMetric(
			unavailableGaugeDesc,
			metrics.GaugeValue,
			gaugeValue,
			apiServiceName,
		)
	}
}

// SetAPIServiceAvailable sets the given apiservice availability gauge to available.
func (c *availabilityCollector) SetAPIServiceAvailable(apiServiceKey string) {
	c.setAPIServiceAvailability(apiServiceKey, true)
}

// SetAPIServiceUnavailable sets the given apiservice availability gauge to unavailable.
func (c *availabilityCollector) SetAPIServiceUnavailable(apiServiceKey string) {
	c.setAPIServiceAvailability(apiServiceKey, false)
}

func (c *availabilityCollector) setAPIServiceAvailability(apiServiceKey string, availability bool) {
	c.mtx.Lock()
	defer c.mtx.Unlock()

	c.availabilities[apiServiceKey] = availability
}

// ForgetAPIService removes the availability gauge of the given apiservice.
func (c *availabilityCollector) ForgetAPIService(apiServiceKey string) {
	c.mtx.Lock()
	defer c.mtx.Unlock()

	delete(c.availabilities, apiServiceKey)
}
