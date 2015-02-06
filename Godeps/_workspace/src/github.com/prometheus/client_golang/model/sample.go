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

// Sample is a sample value with a timestamp and a metric.
type Sample struct {
	Metric    Metric
	Value     SampleValue
	Timestamp Timestamp
}

// Equal compares first the metrics, then the timestamp, then the value.
func (s *Sample) Equal(o *Sample) bool {
	if s == o {
		return true
	}

	if !s.Metric.Equal(o.Metric) {
		return false
	}
	if !s.Timestamp.Equal(o.Timestamp) {
		return false
	}
	if !s.Value.Equal(o.Value) {
		return false
	}

	return true
}

// Samples is a sortable Sample slice. It implements sort.Interface.
type Samples []*Sample

func (s Samples) Len() int {
	return len(s)
}

// Less compares first the metrics, then the timestamp.
func (s Samples) Less(i, j int) bool {
	switch {
	case s[i].Metric.Before(s[j].Metric):
		return true
	case s[j].Metric.Before(s[i].Metric):
		return false
	case s[i].Timestamp.Before(s[j].Timestamp):
		return true
	default:
		return false
	}
}

func (s Samples) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// Equal compares two sets of samples and returns true if they are equal.
func (s Samples) Equal(o Samples) bool {
	if len(s) != len(o) {
		return false
	}

	for i, sample := range s {
		if !sample.Equal(o[i]) {
			return false
		}
	}
	return true
}
