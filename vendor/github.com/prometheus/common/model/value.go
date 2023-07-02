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
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

var (
	// ZeroSample is the pseudo zero-value of Sample used to signal a
	// non-existing sample. It is a Sample with timestamp Earliest, value 0.0,
	// and metric nil. Note that the natural zero value of Sample has a timestamp
	// of 0, which is possible to appear in a real Sample and thus not suitable
	// to signal a non-existing Sample.
	ZeroSample = Sample{Timestamp: Earliest}
)

// Sample is a sample pair associated with a metric. A single sample must either
// define Value or Histogram but not both. Histogram == nil implies the Value
// field is used, otherwise it should be ignored.
type Sample struct {
	Metric    Metric           `json:"metric"`
	Value     SampleValue      `json:"value"`
	Timestamp Time             `json:"timestamp"`
	Histogram *SampleHistogram `json:"histogram"`
}

// Equal compares first the metrics, then the timestamp, then the value. The
// semantics of value equality is defined by SampleValue.Equal.
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
	if s.Histogram != nil {
		return s.Histogram.Equal(o.Histogram)
	}
	return s.Value.Equal(o.Value)
}

func (s Sample) String() string {
	if s.Histogram != nil {
		return fmt.Sprintf("%s => %s", s.Metric, SampleHistogramPair{
			Timestamp: s.Timestamp,
			Histogram: s.Histogram,
		})
	}
	return fmt.Sprintf("%s => %s", s.Metric, SamplePair{
		Timestamp: s.Timestamp,
		Value:     s.Value,
	})
}

// MarshalJSON implements json.Marshaler.
func (s Sample) MarshalJSON() ([]byte, error) {
	if s.Histogram != nil {
		v := struct {
			Metric    Metric              `json:"metric"`
			Histogram SampleHistogramPair `json:"histogram"`
		}{
			Metric: s.Metric,
			Histogram: SampleHistogramPair{
				Timestamp: s.Timestamp,
				Histogram: s.Histogram,
			},
		}
		return json.Marshal(&v)
	}
	v := struct {
		Metric Metric     `json:"metric"`
		Value  SamplePair `json:"value"`
	}{
		Metric: s.Metric,
		Value: SamplePair{
			Timestamp: s.Timestamp,
			Value:     s.Value,
		},
	}
	return json.Marshal(&v)
}

// UnmarshalJSON implements json.Unmarshaler.
func (s *Sample) UnmarshalJSON(b []byte) error {
	v := struct {
		Metric    Metric              `json:"metric"`
		Value     SamplePair          `json:"value"`
		Histogram SampleHistogramPair `json:"histogram"`
	}{
		Metric: s.Metric,
		Value: SamplePair{
			Timestamp: s.Timestamp,
			Value:     s.Value,
		},
		Histogram: SampleHistogramPair{
			Timestamp: s.Timestamp,
			Histogram: s.Histogram,
		},
	}

	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}

	s.Metric = v.Metric
	if v.Histogram.Histogram != nil {
		s.Timestamp = v.Histogram.Timestamp
		s.Histogram = v.Histogram.Histogram
	} else {
		s.Timestamp = v.Value.Timestamp
		s.Value = v.Value.Value
	}

	return nil
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

// SampleStream is a stream of Values belonging to an attached COWMetric.
type SampleStream struct {
	Metric     Metric                `json:"metric"`
	Values     []SamplePair          `json:"values"`
	Histograms []SampleHistogramPair `json:"histograms"`
}

func (ss SampleStream) String() string {
	valuesLength := len(ss.Values)
	vals := make([]string, valuesLength+len(ss.Histograms))
	for i, v := range ss.Values {
		vals[i] = v.String()
	}
	for i, v := range ss.Histograms {
		vals[i+valuesLength] = v.String()
	}
	return fmt.Sprintf("%s =>\n%s", ss.Metric, strings.Join(vals, "\n"))
}

func (ss SampleStream) MarshalJSON() ([]byte, error) {
	if len(ss.Histograms) > 0 && len(ss.Values) > 0 {
		v := struct {
			Metric     Metric                `json:"metric"`
			Values     []SamplePair          `json:"values"`
			Histograms []SampleHistogramPair `json:"histograms"`
		}{
			Metric:     ss.Metric,
			Values:     ss.Values,
			Histograms: ss.Histograms,
		}
		return json.Marshal(&v)
	} else if len(ss.Histograms) > 0 {
		v := struct {
			Metric     Metric                `json:"metric"`
			Histograms []SampleHistogramPair `json:"histograms"`
		}{
			Metric:     ss.Metric,
			Histograms: ss.Histograms,
		}
		return json.Marshal(&v)
	} else {
		v := struct {
			Metric Metric       `json:"metric"`
			Values []SamplePair `json:"values"`
		}{
			Metric: ss.Metric,
			Values: ss.Values,
		}
		return json.Marshal(&v)
	}
}

func (ss *SampleStream) UnmarshalJSON(b []byte) error {
	v := struct {
		Metric     Metric                `json:"metric"`
		Values     []SamplePair          `json:"values"`
		Histograms []SampleHistogramPair `json:"histograms"`
	}{
		Metric:     ss.Metric,
		Values:     ss.Values,
		Histograms: ss.Histograms,
	}

	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}

	ss.Metric = v.Metric
	ss.Values = v.Values
	ss.Histograms = v.Histograms

	return nil
}

// Scalar is a scalar value evaluated at the set timestamp.
type Scalar struct {
	Value     SampleValue `json:"value"`
	Timestamp Time        `json:"timestamp"`
}

func (s Scalar) String() string {
	return fmt.Sprintf("scalar: %v @[%v]", s.Value, s.Timestamp)
}

// MarshalJSON implements json.Marshaler.
func (s Scalar) MarshalJSON() ([]byte, error) {
	v := strconv.FormatFloat(float64(s.Value), 'f', -1, 64)
	return json.Marshal([...]interface{}{s.Timestamp, string(v)})
}

// UnmarshalJSON implements json.Unmarshaler.
func (s *Scalar) UnmarshalJSON(b []byte) error {
	var f string
	v := [...]interface{}{&s.Timestamp, &f}

	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}

	value, err := strconv.ParseFloat(f, 64)
	if err != nil {
		return fmt.Errorf("error parsing sample value: %s", err)
	}
	s.Value = SampleValue(value)
	return nil
}

// String is a string value evaluated at the set timestamp.
type String struct {
	Value     string `json:"value"`
	Timestamp Time   `json:"timestamp"`
}

func (s *String) String() string {
	return s.Value
}

// MarshalJSON implements json.Marshaler.
func (s String) MarshalJSON() ([]byte, error) {
	return json.Marshal([]interface{}{s.Timestamp, s.Value})
}

// UnmarshalJSON implements json.Unmarshaler.
func (s *String) UnmarshalJSON(b []byte) error {
	v := [...]interface{}{&s.Timestamp, &s.Value}
	return json.Unmarshal(b, &v)
}

// Vector is basically only an alias for Samples, but the
// contract is that in a Vector, all Samples have the same timestamp.
type Vector []*Sample

func (vec Vector) String() string {
	entries := make([]string, len(vec))
	for i, s := range vec {
		entries[i] = s.String()
	}
	return strings.Join(entries, "\n")
}

func (vec Vector) Len() int      { return len(vec) }
func (vec Vector) Swap(i, j int) { vec[i], vec[j] = vec[j], vec[i] }

// Less compares first the metrics, then the timestamp.
func (vec Vector) Less(i, j int) bool {
	switch {
	case vec[i].Metric.Before(vec[j].Metric):
		return true
	case vec[j].Metric.Before(vec[i].Metric):
		return false
	case vec[i].Timestamp.Before(vec[j].Timestamp):
		return true
	default:
		return false
	}
}

// Equal compares two sets of samples and returns true if they are equal.
func (vec Vector) Equal(o Vector) bool {
	if len(vec) != len(o) {
		return false
	}

	for i, sample := range vec {
		if !sample.Equal(o[i]) {
			return false
		}
	}
	return true
}

// Matrix is a list of time series.
type Matrix []*SampleStream

func (m Matrix) Len() int           { return len(m) }
func (m Matrix) Less(i, j int) bool { return m[i].Metric.Before(m[j].Metric) }
func (m Matrix) Swap(i, j int)      { m[i], m[j] = m[j], m[i] }

func (mat Matrix) String() string {
	matCp := make(Matrix, len(mat))
	copy(matCp, mat)
	sort.Sort(matCp)

	strs := make([]string, len(matCp))

	for i, ss := range matCp {
		strs[i] = ss.String()
	}

	return strings.Join(strs, "\n")
}
