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

package feature

import (
	"context"
	"fmt"

	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var (
	// featureEnabled is a Prometheus Gauge metrics used for recording the enablement of a k8s feature.
	featureEnabled = k8smetrics.NewGaugeVec(
		&k8smetrics.GaugeOpts{
			Namespace:      "k8s",
			Name:           "feature_enabled",
			Help:           "This metric records the result of whether a feature is enabled.",
			StabilityLevel: k8smetrics.ALPHA,
		},
		[]string{"name", "enabled"},
	)
)

func init() {
	legacyregistry.MustRegister(featureEnabled)
}

func ResetFeatureEnabledMetric() {
	featureEnabled.Reset()
}

func RecordFeatureEnabled(ctx context.Context, name string, enabled bool) error {
	featureEnabled.WithContext(ctx).WithLabelValues(name, fmt.Sprintf("%v", enabled)).Set(1)
	return nil
}
