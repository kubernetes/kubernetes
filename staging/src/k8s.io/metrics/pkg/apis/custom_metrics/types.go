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
	"k8s.io/client-go/pkg/api"
)

// a list of values for a given metric for some set of objects
type MetricValueList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`

	// the value of the metric across the described objects
	Items []MetricValue `json:"items"`
}

// a metric value for some object
type MetricValue struct {
	metav1.TypeMeta `json:",inline"`

	// a reference to the described object
	DescribedObject api.ObjectReference `json:"describedObject"`

	// the name of the metric
	MetricName string `json:"metricName"`

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
