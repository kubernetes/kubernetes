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
	"math"
	"sort"
	"strconv"
	"strings"
)

// A SampleValue is a representation of a value for a given sample at a given
// time.
type SampleValue float64

// MarshalJSON implements json.Marshaler.
func (v SampleValue) MarshalJSON() ([]byte, error) {
	return json.Marshal(v.String())
}

// UnmarshalJSON implements json.Unmarshaler.
func (v *SampleValue) UnmarshalJSON(b []byte) error {
	if len(b) < 2 || b[0] != '"' || b[len(b)-1] != '"' {
		return fmt.Errorf("sample value must be a quoted string")
	}
	f, err := strconv.ParseFloat(string(b[1:len(b)-1]), 64)
	if err != nil {
		return err
	}
	*v = SampleValue(f)
	return nil
}

// Equal returns true if the value of v and o is equal or if both are NaN. Note
// that v==o is false if both are NaN. If you want the conventional float
// behavior, use == to compare two SampleValues.
func (v SampleValue) Equal(o SampleValue) bool {
	if v == o {
		return true
	}
	return math.IsNaN(float64(v)) && math.IsNaN(float64(o))
}

func (v SampleValue) String() string {
	return strconv.FormatFloat(float64(v), 'f', -1, 64)
}

// SamplePair pairs a SampleValue with a Timestamp.
type SamplePair struct {
	Timestamp Time
	Value     SampleValue
}

// MarshalJSON implements json.Marshaler.
func (s SamplePair) MarshalJSON() ([]byte, error) {
	t, err := json.Marshal(s.Timestamp)
	if err != nil {
		return nil, err
	}
	v, err := json.Marshal(s.Value)
	if err != nil {
		return nil, err
	}
	return []byte(fmt.Sprintf("[%s,%s]", t, v)), nil
}

// UnmarshalJSON implements json.Unmarshaler.
func (s *SamplePair) UnmarshalJSON(b []byte) error {
	v := [...]json.Unmarshaler{&s.Timestamp, &s.Value}
	return json.Unmarshal(b, &v)
}

// Equal returns true if this SamplePair and o have equal Values and equal
// Timestamps. The sematics of Value equality is defined by SampleValue.Equal.
func (s *SamplePair) Equal(o *SamplePair) bool {
	return s == o || (s.Value.Equal(o.Value) && s.Timestamp.Equal(o.Timestamp))
}

func (s SamplePair) String() string {
	return fmt.Sprintf("%s @[%s]", s.Value, s.Timestamp)
}

// Sample is a sample pair associated with a metric.
type Sample struct {
	Metric    Metric      `json:"metric"`
	Value     SampleValue `json:"value"`
	Timestamp Time        `json:"timestamp"`
}

// Equal compares first the metrics, then the timestamp, then the value. The
// sematics of value equality is defined by SampleValue.Equal.
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
	if s.Value.Equal(o.Value) {
		return false
	}

	return true
}

func (s Sample) String() string {
	return fmt.Sprintf("%s => %s", s.Metric, SamplePair{
		Timestamp: s.Timestamp,
		Value:     s.Value,
	})
}

// MarshalJSON implements json.Marshaler.
func (s Sample) MarshalJSON() ([]byte, error) {
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
		Metric Metric     `json:"metric"`
		Value  SamplePair `json:"value"`
	}{
		Metric: s.Metric,
		Value: SamplePair{
			Timestamp: s.Timestamp,
			Value:     s.Value,
		},
	}

	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}

	s.Metric = v.Metric
	s.Timestamp = v.Value.Timestamp
	s.Value = v.Value.Value

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
	Metric Metric       `json:"metric"`
	Values []SamplePair `json:"values"`
}

func (ss SampleStream) String() string {
	vals := make([]string, len(ss.Values))
	for i, v := range ss.Values {
		vals[i] = v.String()
	}
	return fmt.Sprintf("%s =>\n%s", ss.Metric, strings.Join(vals, "\n"))
}

// Value is a generic interface for values resulting from a query evaluation.
type Value interface {
	Type() ValueType
	String() string
}

func (Matrix) Type() ValueType  { return ValMatrix }
func (Vector) Type() ValueType  { return ValVector }
func (*Scalar) Type() ValueType { return ValScalar }
func (*String) Type() ValueType { return ValString }

type ValueType int

const (
	ValNone ValueType = iota
	ValScalar
	ValVector
	ValMatrix
	ValString
)

// MarshalJSON implements json.Marshaler.
func (et ValueType) MarshalJSON() ([]byte, error) {
	return json.Marshal(et.String())
}

func (et *ValueType) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	switch s {
	case "<ValNone>":
		*et = ValNone
	case "scalar":
		*et = ValScalar
	case "vector":
		*et = ValVector
	case "matrix":
		*et = ValMatrix
	case "string":
		*et = ValString
	default:
		return fmt.Errorf("unknown value type %q", s)
	}
	return nil
}

func (e ValueType) String() string {
	switch e {
	case ValNone:
		return "<ValNone>"
	case ValScalar:
		return "scalar"
	case ValVector:
		return "vector"
	case ValMatrix:
		return "matrix"
	case ValString:
		return "string"
	}
	panic("ValueType.String: unhandled value type")
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
