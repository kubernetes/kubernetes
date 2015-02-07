// Copyright 2013 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"strings"
)

const (
	// ExporterLabelPrefix is the label name prefix to prepend if a
	// synthetic label is already present in the exported metrics.
	ExporterLabelPrefix LabelName = "exporter_"

	// MetricNameLabel is the label name indicating the metric name of a
	// timeseries.
	MetricNameLabel LabelName = "__name__"

	// ReservedLabelPrefix is a prefix which is not legal in user-supplied
	// label names.
	ReservedLabelPrefix = "__"

	// JobLabel is the label name indicating the job from which a timeseries
	// was scraped.
	JobLabel LabelName = "job"
)

// A LabelName is a key for a LabelSet or Metric.  It has a value associated
// therewith.
type LabelName string

// LabelNames is a sortable LabelName slice. In implements sort.Interface.
type LabelNames []LabelName

func (l LabelNames) Len() int {
	return len(l)
}

func (l LabelNames) Less(i, j int) bool {
	return l[i] < l[j]
}

func (l LabelNames) Swap(i, j int) {
	l[i], l[j] = l[j], l[i]
}

func (l LabelNames) String() string {
	labelStrings := make([]string, 0, len(l))
	for _, label := range l {
		labelStrings = append(labelStrings, string(label))
	}
	return strings.Join(labelStrings, ", ")
}
