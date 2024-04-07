/*
Copyright 2022 The Kubernetes Authors.

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

package slis

import (
	"context"
	k8smetrics "k8s.io/component-base/metrics"
)

type HealthcheckStatus string

const (
	Success HealthcheckStatus = "success"
	Error   HealthcheckStatus = "error"
)

type HealthcheckType string

var (
	// healthcheck is a Prometheus Gauge metrics used for recording the results of a k8s healthcheck.
	healthcheck = k8smetrics.NewGaugeVec(
		&k8smetrics.GaugeOpts{
			Namespace:      "kubernetes",
			Name:           "healthcheck",
			Help:           "This metric records the result of a single healthcheck.",
			StabilityLevel: k8smetrics.STABLE,
		},
		[]string{"name", "type"},
	)

	// healthchecksTotal is a Prometheus Counter metrics used for counting the results of a k8s healthcheck.
	healthchecksTotal = k8smetrics.NewCounterVec(
		&k8smetrics.CounterOpts{
			Namespace:      "kubernetes",
			Name:           "healthchecks_total",
			Help:           "This metric records the results of all healthcheck.",
			StabilityLevel: k8smetrics.STABLE,
		},
		[]string{"name", "type", "status"},
	)
)

func Register(registry k8smetrics.KubeRegistry) {
	registry.Register(healthcheck)
	registry.Register(healthchecksTotal)
}

func ResetHealthMetrics() {
	healthcheck.Reset()
	healthchecksTotal.Reset()
}

func ObserveHealthcheck(ctx context.Context, name string, healthcheckType string, status HealthcheckStatus) error {
	if status == Success {
		healthcheck.WithContext(ctx).WithLabelValues(name, healthcheckType).Set(1)
	} else {
		healthcheck.WithContext(ctx).WithLabelValues(name, healthcheckType).Set(0)
	}

	healthchecksTotal.WithContext(ctx).WithLabelValues(name, healthcheckType, string(status)).Inc()
	return nil
}
