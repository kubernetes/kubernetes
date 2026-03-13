/*
Copyright 2016 The Kubernetes Authors.

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

package perftype

// TODO(random-liu): Replace this with prometheus' data model.

// The following performance data structures are generalized and well-formatted.
// They can be pretty printed in json format and be analyzed by other performance
// analyzing tools, such as Perfdash (k8s.io/contrib/perfdash).

// DataItem is the data point.
type DataItem struct {
	// Data is a map from bucket to real data point (e.g. "Perc90" -> 23.5). Notice
	// that all data items with the same label combination should have the same buckets.
	Data map[string]float64 `json:"data"`
	// Unit is the data unit. Notice that all data items with the same label combination
	// should have the same unit.
	Unit string `json:"unit"`
	// Labels is the labels of the data item.
	Labels map[string]string `json:"labels,omitempty"`
}

// PerfData contains all data items generated in current test.
type PerfData struct {
	// Version is the version of the metrics. The metrics consumer could use the version
	// to detect metrics version change and decide what version to support.
	Version   string     `json:"version"`
	DataItems []DataItem `json:"dataItems"`
	// Labels is the labels of the dataset.
	Labels map[string]string `json:"labels,omitempty"`
}

// PerfResultTag is the prefix of generated perfdata. Analyzing tools can find the perf result
// with this tag.
const PerfResultTag = "[Result:Performance]"

// PerfResultEnd is the end of generated perfdata. Analyzing tools can find the end of the perf
// result with this tag.
const PerfResultEnd = "[Finish:Performance]"
