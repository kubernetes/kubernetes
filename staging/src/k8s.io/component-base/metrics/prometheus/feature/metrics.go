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

	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var (
	// featureInfo is a Prometheus Gauge metrics used for recording the enablement of a k8s feature.
	featureInfo = k8smetrics.NewGaugeVec(
		&k8smetrics.GaugeOpts{
			Namespace:      "kubernetes",
			Name:           "feature_enabled",
			Help:           "This metric records the data about the stage and enablement of a k8s feature.",
			StabilityLevel: k8smetrics.ALPHA,
		},
		[]string{"name", "stage"},
	)
)

func init() {
	legacyregistry.MustRegister(featureInfo)
}

func ResetFeatureInfoMetric() {
	featureInfo.Reset()
}

func RecordFeatureInfo(ctx context.Context, name string, stage string, enabled bool) {
	value := 0.0
	if enabled {
		value = 1.0
	}
	featureInfo.WithContext(ctx).WithLabelValues(name, stage).Set(value)
}
