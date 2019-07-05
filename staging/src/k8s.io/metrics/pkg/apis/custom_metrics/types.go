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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

type MetricIdentifier struct {
	// name is the name of the given metric
	Name string
	// selector represents the label selector that could be used to select
	// this metric, and will generally just be the selector passed in to
	// the query used to fetch this metric.
	// +optional
	Selector *metav1.LabelSelector
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// a list of values for a given metric for some set of objects
type MetricValueList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`

	// the value of the metric across the described objects
	Items []MetricValue `json:"items"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// a metric value for some object
type MetricValue struct {
	metav1.TypeMeta `json:",inline"`

	// a reference to the described object
	DescribedObject ObjectReference `json:"describedObject"`

	Metric MetricIdentifier

	// indicates the time at which the metrics were produced
	Timestamp metav1.Time `json:"timestamp"`

	// indicates the window ([Timestamp-Window, Timestamp]) from
	// which these metrics were calculated, when returning rate
	// metrics calculated from cumulative metrics (or zero for
	// non-calculated instantaneous metrics).
	WindowSeconds *int64 `json:"window,omitempty"`

	// the value of the metric for this
	Value resource.Quantity `json:"value"`
}

// allObjects is a wildcard used to select metrics
// for all objects matching the given label selector
const AllObjects = "*"

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// MetricListOptions is used to select metrics by their label selectors
type MetricListOptions struct {
	metav1.TypeMeta `json:",inline"`

	// A selector to restrict the list of returned objects by their labels.
	// Defaults to everything.
	// +optional
	LabelSelector string `json:"labelSelector,omitempty" protobuf:"bytes,1,opt,name=labelSelector"`

	// A selector to restrict the list of returned metrics by their labels
	// +optional
	MetricLabelSelector string `json:"metricLabelSelector,omitempty" protobuf:"bytes,2,opt,name=metricLabelSelector"`
}

// NOTE: ObjectReference is copied from k8s.io/kubernetes/pkg/api/types.go. We
// cannot depend on k8s.io/kubernetes/pkg/api because that creates cyclic
// dependency between k8s.io/metrics and k8s.io/kubernetes. We cannot depend on
// k8s.io/client-go/pkg/api because the package is going to be deprecated soon.
// There is no need to keep it an exact copy. Each repo can define its own
// internal objects.

// ObjectReference contains enough information to let you inspect or modify the referred object.
type ObjectReference struct {
	Kind            string
	Namespace       string
	Name            string
	UID             types.UID
	APIVersion      string
	ResourceVersion string
	FieldPath       string
}
