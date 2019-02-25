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

package external_metrics

import (
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/metrics/pkg/apis/external_metrics/v1beta1"
)

// ExternalMetricsClient is a client for fetching external metrics.
type ExternalMetricsClient interface {
	NamespacedMetricsGetter
}

// NamespacedMetricsGetter provides access to an interface for fetching
// metrics in a particular namespace.
type NamespacedMetricsGetter interface {
	NamespacedMetrics(namespace string) MetricsInterface
}

// MetricsInterface provides access to external metrics.
type MetricsInterface interface {
	// List fetches the metric for the given namespace that maches the given
	// metricSelector.
	List(metricName string, metricSelector labels.Selector) (*v1beta1.ExternalMetricValueList, error)
}
