// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package types

import (
	"time"
)

// Timeseries represents a set of metrics for the same target object
// (typically a container).
type Timeseries struct {
	// Map of metric names to their values.
	Metrics map[string][]Point `json:"metrics"`

	// Common labels for all metrics.
	Labels map[string]string `json:"labels,omitempty"`
}

// Point represent a metric value.
type Point struct {
	// The start and end time for which this data is representative.
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`

	// Labels specific to this data point.
	Labels map[string]string `json:"labels,omitempty"`

	// The value of the metric.
	Value interface{} `json:"value"`
}

// TimeseriesSchema represents all the metrics and labels.
type TimeseriesSchema struct {
	// All the metrics handled by heapster.
	Metrics []MetricDescriptor `json:"metrics,omitempty"`
	// Labels that are common to all metrics.
	CommonLabels []LabelDescriptor `json:"common_labels,omitempty"`
	// Labels that are present only for containers in pods.
	// A container metric belongs to a pod is "pod_name" label is set.
	PodLabels []LabelDescriptor `json:"pod_labels,omitempty"`
}

type MetricDescriptor struct {
	// The unique name of the metric.
	Name string `json:"name,omitempty"`

	// Description of the metric.
	Description string `json:"description,omitempty"`

	// Descriptor of the labels specific to this metric.
	Labels []LabelDescriptor `json:"labels,omitempty"`

	// Type and value of metric data.
	Type string `json:"type,omitempty"`

	// The type of value returned as part of this metric.
	ValueType string `json:"value_type,omitempty"`

	// The units of the value returned as part of this metric.
	Units string `json:"units,omitempty"`
}

type LabelDescriptor struct {
	// Key to use for the label.
	Key string `json:"key,omitempty"`

	// Description of the label.
	Description string `json:"description,omitempty"`
}
