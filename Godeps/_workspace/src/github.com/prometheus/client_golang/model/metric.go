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
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"sort"
	"strings"
)

var separator = []byte{0}

// A Metric is similar to a LabelSet, but the key difference is that a Metric is
// a singleton and refers to one and only one stream of samples.
type Metric map[LabelName]LabelValue

// Equal compares the fingerprints of both metrics.
func (m Metric) Equal(o Metric) bool {
	return m.Fingerprint().Equal(o.Fingerprint())
}

// Before compares the fingerprints of both metrics.
func (m Metric) Before(o Metric) bool {
	return m.Fingerprint().Less(o.Fingerprint())
}

// String implements Stringer.
func (m Metric) String() string {
	metricName, hasName := m[MetricNameLabel]
	numLabels := len(m) - 1
	if !hasName {
		numLabels = len(m)
	}
	labelStrings := make([]string, 0, numLabels)
	for label, value := range m {
		if label != MetricNameLabel {
			labelStrings = append(labelStrings, fmt.Sprintf("%s=%q", label, value))
		}
	}

	switch numLabels {
	case 0:
		if hasName {
			return string(metricName)
		}
		return "{}"
	default:
		sort.Strings(labelStrings)
		return fmt.Sprintf("%s{%s}", metricName, strings.Join(labelStrings, ", "))
	}
}

// Fingerprint returns a Metric's Fingerprint.
func (m Metric) Fingerprint() Fingerprint {
	labelNames := make([]string, 0, len(m))
	maxLength := 0

	for labelName, labelValue := range m {
		labelNames = append(labelNames, string(labelName))
		if len(labelName) > maxLength {
			maxLength = len(labelName)
		}
		if len(labelValue) > maxLength {
			maxLength = len(labelValue)
		}
	}

	sort.Strings(labelNames)

	summer := fnv.New64a()
	buf := make([]byte, maxLength)

	for _, labelName := range labelNames {
		labelValue := m[LabelName(labelName)]

		copy(buf, labelName)
		summer.Write(buf[:len(labelName)])

		summer.Write(separator)

		copy(buf, labelValue)
		summer.Write(buf[:len(labelValue)])
	}

	return Fingerprint(binary.LittleEndian.Uint64(summer.Sum(nil)))
}

// Clone returns a copy of the Metric.
func (m Metric) Clone() Metric {
	clone := Metric{}
	for k, v := range m {
		clone[k] = v
	}
	return clone
}

// MergeFromLabelSet merges a label set into this Metric, prefixing a collision
// prefix to the label names merged from the label set where required.
func (m Metric) MergeFromLabelSet(labels LabelSet, collisionPrefix LabelName) {
	for k, v := range labels {
		if collisionPrefix != "" {
			for {
				if _, exists := m[k]; !exists {
					break
				}
				k = collisionPrefix + k
			}
		}

		m[k] = v
	}
}

// COWMetric wraps a Metric to enable copy-on-write access patterns.
type COWMetric struct {
	Copied bool
	Metric Metric
}

// Set sets a label name in the wrapped Metric to a given value and copies the
// Metric initially, if it is not already a copy.
func (m COWMetric) Set(ln LabelName, lv LabelValue) {
	m.doCOW()
	m.Metric[ln] = lv
}

// Delete deletes a given label name from the wrapped Metric and copies the
// Metric initially, if it is not already a copy.
func (m *COWMetric) Delete(ln LabelName) {
	m.doCOW()
	delete(m.Metric, ln)
}

// doCOW copies the underlying Metric if it is not already a copy.
func (m *COWMetric) doCOW() {
	if !m.Copied {
		m.Metric = m.Metric.Clone()
		m.Copied = true
	}
}

// String implements fmt.Stringer.
func (m COWMetric) String() string {
	return m.Metric.String()
}

// MarshalJSON implements json.Marshaler.
func (m COWMetric) MarshalJSON() ([]byte, error) {
	return json.Marshal(m.Metric)
}
