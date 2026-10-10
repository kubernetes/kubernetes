/*
Copyright The Kubernetes Authors.

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

package fakemetricsserver

import (
	"fmt"
	"sync"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
	metricsv1 "k8s.io/metrics/pkg/apis/metrics/v1"
	metricsv1beta1 "k8s.io/metrics/pkg/apis/metrics/v1beta1"
)

type metricProvider struct {
	lock    sync.RWMutex
	metrics map[string][]metricsv1.PodMetrics
}

func newMetricProvider() *metricProvider {
	return &metricProvider{
		metrics: make(map[string][]metricsv1.PodMetrics),
	}
}

func (p *metricProvider) replace(metrics []metricsv1.PodMetrics) {
	byNamespace := make(map[string][]metricsv1.PodMetrics)

	for i := range metrics {
		metric := metrics[i].DeepCopy()
		byNamespace[metric.Namespace] = append(
			byNamespace[metric.Namespace],
			*metric,
		)
	}

	p.lock.Lock()
	defer p.lock.Unlock()
	p.metrics = byNamespace
}

func (p *metricProvider) list(namespace string, selector labels.Selector) []metricsv1.PodMetrics {
	p.lock.RLock()
	defer p.lock.RUnlock()

	stored := p.metrics[namespace]
	items := make([]metricsv1.PodMetrics, 0, len(stored))

	for i := range stored {
		if !selector.Matches(labels.Set(stored[i].Labels)) {
			continue
		}
		items = append(items, *stored[i].DeepCopy())
	}

	return items
}

func (p *metricProvider) listV1(namespace string, selector labels.Selector) metricsv1.PodMetricsList {
	return metricsv1.PodMetricsList{
		TypeMeta: metav1.TypeMeta{
			APIVersion: metricsv1.SchemeGroupVersion.String(),
			Kind:       "PodMetricsList",
		},
		Items: p.list(namespace, selector),
	}
}

func (p *metricProvider) listV1beta1(namespace string, selector labels.Selector) (metricsv1beta1.PodMetricsList, error) {
	v1Metrics := metricsv1.PodMetricsList{
		Items: p.list(namespace, selector),
	}

	converted, err := convertV1ToV1beta1(&v1Metrics)
	if err != nil {
		return metricsv1beta1.PodMetricsList{}, err
	}
	return *converted, nil
}

func convertV1ToV1beta1(in *metricsv1.PodMetricsList) (*metricsv1beta1.PodMetricsList, error) {
	internal := &metricsapi.PodMetricsList{}
	if err := metricsv1.Convert_v1_PodMetricsList_To_metrics_PodMetricsList(in, internal, nil); err != nil {
		return nil, fmt.Errorf("failed to convert v1 PodMetricsList to internal version: %w", err)
	}

	out := &metricsv1beta1.PodMetricsList{}
	if err := metricsv1beta1.Convert_metrics_PodMetricsList_To_v1beta1_PodMetricsList(internal, out, nil); err != nil {
		return nil, fmt.Errorf("failed to convert PodMetricsList to v1beta1: %w", err)
	}

	out.TypeMeta = metav1.TypeMeta{
		APIVersion: metricsv1beta1.SchemeGroupVersion.String(),
		Kind:       "PodMetricsList",
	}
	return out, nil
}
