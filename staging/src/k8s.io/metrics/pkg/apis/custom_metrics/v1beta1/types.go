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

package v1beta1

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// a list of values for a given metric for some set of objects
type MetricValueList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// the value of the metric across the described objects
	Items []MetricValue `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// a metric value for some object
type MetricValue struct {
	metav1.TypeMeta `json:",inline"`

	// a reference to the described object
	DescribedObject v1.ObjectReference `json:"describedObject" protobuf:"bytes,1,name=describedObject"`

	// the name of the metric
	MetricName string `json:"metricName" protobuf:"bytes,2,name=metricName"`

	// indicates the time at which the metrics were produced
	Timestamp metav1.Time `json:"timestamp" protobuf:"bytes,3,name=timestamp"`

	// indicates the window ([Timestamp-Window, Timestamp]) from
	// which these metrics were calculated, when returning rate
	// metrics calculated from cumulative metrics (or zero for
	// non-calculated instantaneous metrics).
	WindowSeconds *int64 `json:"window,omitempty" protobuf:"bytes,4,opt,name=window"`

	// the value of the metric for this
	Value resource.Quantity `json:"value" protobuf:"bytes,5,name=value"`
}

// allObjects is a wildcard used to select metrics
// for all objects matching the given label selector
const AllObjects = "*"
