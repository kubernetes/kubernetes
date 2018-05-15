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

package metrics

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
)

// Resettable allows for a reset of a value(s).
type Resettable interface {
	//Reset the value of a metric(s).
	Reset()
}

// ResettableCollector is the interface implmented by any
// Prometheus metric, which wants to have its value reset.
type ResettableCollector interface {
	prometheus.Collector
	Resettable
}

// Group is the interface which describes a group
// containing one or more metrics.
type Group interface {
	// GetMetrics returns all metrics in that group.
	GetMetrics() []ResettableCollector
}

// Store is the interface which allows one or more metrics
// to be registred to a Prometheus registry and have their value
// reset.
type Store interface {
	Resettable

	// Register all associated metrics to a Prometheus registry.
	Register()
}

type defaultStore struct {
	metrics         []ResettableCollector
	registerMetrics sync.Once
	register        prometheus.Registerer
}

type metricsGroup struct {
	metrics []ResettableCollector
}

// Register all associated metrics to a Prometheus registry.
// Can be called only once.
func (o *defaultStore) Register() {
	o.registerMetrics.Do(func() {
		for _, metric := range o.metrics {
			o.register.MustRegister(metric)
		}
	})
}

// Reset all associated metrics to their initial values.
func (o *defaultStore) Reset() {
	for _, metric := range o.metrics {
		metric.Reset()
	}
}

// NewStore creates a new default store, which registers
// metrics from a group to a Prometheus registry.
func NewStore(r prometheus.Registerer, m ...Group) Store {
	if r == nil {
		r = prometheus.DefaultRegisterer
	}
	metrics := []ResettableCollector{}
	for _, mg := range m {
		metrics = append(metrics, mg.GetMetrics()...)
	}
	return &defaultStore{
		metrics:         metrics,
		registerMetrics: sync.Once{},
		register:        r,
	}
}

func (m *metricsGroup) GetMetrics() []ResettableCollector {
	return m.metrics
}

// NewGroup creates a new group with multiple metrics.
func NewGroup(m ...ResettableCollector) Group {
	return &metricsGroup{
		metrics: m,
	}
}
