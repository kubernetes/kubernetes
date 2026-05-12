/*
Copyright 2025 The Kubernetes Authors.

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

package compatversion

import (
	"context"

	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var (
	// compatVersionInfo is a Prometheus Gauge metrics used for recording the current emulation and binary version of a component.
	compatVersionInfo = k8smetrics.NewGaugeVec(
		&k8smetrics.GaugeOpts{
			Name: "version_info",
			Help: "Provides the compatibility version info of the component. " +
				"The component label is the name of the component, usually kube, " +
				"but is relevant for aggregated-apiservers.",
			StabilityLevel: k8smetrics.ALPHA,
		},
		[]string{"component", "binary", "emulation", "min_compat"},
	)
)

func ResetCompatVersionInfoMetric() {
	compatVersionInfo.Reset()
}

func RecordCompatVersionInfo(ctx context.Context, component, binary, emulation, minCompat string) {
	compatVersionInfo.WithContext(ctx).WithLabelValues(component, binary, emulation, minCompat).Set(1)
}

func init() {
	legacyregistry.MustRegister(compatVersionInfo)
}
