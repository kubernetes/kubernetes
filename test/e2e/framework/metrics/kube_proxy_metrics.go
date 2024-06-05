/*
Copyright 2024 The Kubernetes Authors.

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
	"k8s.io/component-base/metrics/testutil"
)

// KubeProxyMetrics is metrics for kube-proxy
type KubeProxyMetrics testutil.Metrics

// GetCounterMetricValue returns value for metric type counter.
func (m *KubeProxyMetrics) GetCounterMetricValue(metricName string) float64 {
	return float64(testutil.Metrics(*m)[metricName][0].Value)
}

func newKubeProxyMetricsMetrics() KubeProxyMetrics {
	result := testutil.NewMetrics()
	return KubeProxyMetrics(result)
}

func parseKubeProxyMetrics(data string) (KubeProxyMetrics, error) {
	result := newKubeProxyMetricsMetrics()
	if err := testutil.ParseMetrics(data, (*testutil.Metrics)(&result)); err != nil {
		return KubeProxyMetrics{}, err
	}
	return result, nil
}
