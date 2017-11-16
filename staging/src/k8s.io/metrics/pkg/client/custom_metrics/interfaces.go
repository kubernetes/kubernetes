/*
Copyright 2017 The Kubernetes Authors.

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

package custom_metrics

import (
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/metrics/pkg/apis/custom_metrics/v1beta1"
)

// CustomMetricsClient is a client for fetching metrics
// describing both root-scoped and namespaced resources.
type CustomMetricsClient interface {
	RootScopedMetricsGetter
	NamespacedMetricsGetter
}

// RootScopedMetricsGetter provides access to an interface for fetching
// metrics describing root-scoped objects.  Note that metrics describing
// a namespace are simply considered a special case of root-scoped metrics.
type RootScopedMetricsGetter interface {
	RootScopedMetrics() MetricsInterface
}

// NamespacedMetricsGetter provides access to an interface for fetching
// metrics describing resources in a particular namespace.
type NamespacedMetricsGetter interface {
	NamespacedMetrics(namespace string) MetricsInterface
}

// MetricsInterface provides access to metrics describing Kubernetes objects.
type MetricsInterface interface {
	// GetForObject fetchs the given metric describing the given object.
	GetForObject(groupKind schema.GroupKind, name string, metricName string) (*v1beta1.MetricValue, error)

	// GetForObjects fetches the given metric describing all objects of the given
	// type matching the given label selector (or simply all objects of the given type
	// if the selector is nil).
	GetForObjects(groupKind schema.GroupKind, selector labels.Selector, metricName string) (*v1beta1.MetricValueList, error)
}
