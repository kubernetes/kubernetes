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

package health

import (
	"context"
	"errors"

	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

type HealthcheckStatus string

const (
	SUCCESS HealthcheckStatus = "success"
	ERROR   HealthcheckStatus = "error"
	PENDING HealthcheckStatus = "pending"
)

type HealthcheckType string

const (
	LIVEZ   HealthcheckType = "livez"
	READYZ  HealthcheckType = "readyz"
	HEALTHZ HealthcheckType = "healthz"
)

var (
	// healthcheck is a Prometheus Gauge metrics used for recording the results of a k8s healthcheck.
	healthcheck = k8smetrics.NewGaugeVec(
		&k8smetrics.GaugeOpts{
			Name:           "k8s_healthcheck",
			Help:           "This metric records the result of a single health check.",
			StabilityLevel: k8smetrics.ALPHA,
		},
		[]string{"name", "type", "status"},
	)
	statuses  = []HealthcheckStatus{SUCCESS, ERROR, PENDING}
	statusSet = map[HealthcheckStatus]struct{}{SUCCESS: {}, ERROR: {}, PENDING: {}}
	checkSet  = map[HealthcheckType]struct{}{LIVEZ: {}, READYZ: {}, HEALTHZ: {}}
)

func init() {
	legacyregistry.MustRegister(healthcheck)
}

func ResetHealthMetrics() {
	healthcheck.Reset()
}

func ObserveHealthcheck(ctx context.Context, name string, healthcheckType HealthcheckType, status HealthcheckStatus) error {
	if _, ok := statusSet[status]; !ok {
		return errors.New("not a valid healthcheck status")
	}
	if _, ok := checkSet[healthcheckType]; !ok {
		return errors.New("not a valid healthcheck type")
	}
	for _, s := range statuses {
		if status != s {
			healthcheck.WithContext(ctx).WithLabelValues(name, string(healthcheckType), string(s)).Set(0)
		}
	}
	healthcheck.WithContext(ctx).WithLabelValues(name, string(healthcheckType), string(status)).Set(1)
	return nil
}
