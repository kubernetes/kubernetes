// Copyright 2014 The Prometheus Authors
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

package prometheus

import (
	"bytes"
	"fmt"
	"hash"
	"sync"
)

// MetricVec is a Collector to bundle metrics of the same name that
// differ in their label values. MetricVec is usually not used directly but as a
// building block for implementations of vectors of a given metric
// type. GaugeVec, CounterVec, SummaryVec, and UntypedVec are examples already
// provided in this package.
type MetricVec struct {
	mtx      sync.RWMutex // Protects not only children, but also hash and buf.
	children map[uint64]Metric
	desc     *Desc

	// hash is our own hash instance to avoid repeated allocations.
	hash hash.Hash64
	// buf is used to copy string contents into it for hashing,
	// again to avoid allocations.
	buf bytes.Buffer

	newMetric func(labelValues ...string) Metric
}

// Describe implements Collector. The length of the returned slice
// is always one.
func (m *MetricVec) Describe(ch chan<- *Desc) {
	ch <- m.desc
}

// Collect implements Collector.
func (m *MetricVec) Collect(ch chan<- Metric) {
	m.mtx.RLock()
	defer m.mtx.RUnlock()

	for _, metric := range m.children {
		ch <- metric
	}
}

// GetMetricWithLabelValues returns the Metric for the given slice of label
// values (same order as the VariableLabels in Desc). If that combination of
// label values is accessed for the first time, a new Metric is created.
//
// It is possible to call this method without using the returned Metric to only
// create the new Metric but leave it at its start value (e.g. a Summary or
// Histogram without any observations). See also the SummaryVec example.
//
// Keeping the Metric for later use is possible (and should be considered if
// performance is critical), but keep in mind that Reset, DeleteLabelValues and
// Delete can be used to delete the Metric from the MetricVec. In that case, the
// Metric will still exist, but it will not be exported anymore, even if a
// Metric with the same label values is created later. See also the CounterVec
// example.
//
// An error is returned if the number of label values is not the same as the
// number of VariableLabels in Desc.
//
// Note that for more than one label value, this method is prone to mistakes
// caused by an incorrect order of arguments. Consider GetMetricWith(Labels) as
// an alternative to avoid that type of mistake. For higher label numbers, the
// latter has a much more readable (albeit more verbose) syntax, but it comes
// with a performance overhead (for creating and processing the Labels map).
// See also the GaugeVec example.
func (m *MetricVec) GetMetricWithLabelValues(lvs ...string) (Metric, error) {
	m.mtx.Lock()
	defer m.mtx.Unlock()

	h, err := m.hashLabelValues(lvs)
	if err != nil {
		return nil, err
	}
	return m.getOrCreateMetric(h, lvs...), nil
}

// GetMetricWith returns the Metric for the given Labels map (the label names
// must match those of the VariableLabels in Desc). If that label map is
// accessed for the first time, a new Metric is created. Implications of
// creating a Metric without using it and keeping the Metric for later use are
// the same as for GetMetricWithLabelValues.
//
// An error is returned if the number and names of the Labels are inconsistent
// with those of the VariableLabels in Desc.
//
// This method is used for the same purpose as
// GetMetricWithLabelValues(...string). See there for pros and cons of the two
// methods.
func (m *MetricVec) GetMetricWith(labels Labels) (Metric, error) {
	m.mtx.Lock()
	defer m.mtx.Unlock()

	h, err := m.hashLabels(labels)
	if err != nil {
		return nil, err
	}
	lvs := make([]string, len(labels))
	for i, label := range m.desc.variableLabels {
		lvs[i] = labels[label]
	}
	return m.getOrCreateMetric(h, lvs...), nil
}

// WithLabelValues works as GetMetricWithLabelValues, but panics if an error
// occurs. The method allows neat syntax like:
//     httpReqs.WithLabelValues("404", "POST").Inc()
func (m *MetricVec) WithLabelValues(lvs ...string) Metric {
	metric, err := m.GetMetricWithLabelValues(lvs...)
	if err != nil {
		panic(err)
	}
	return metric
}

// With works as GetMetricWith, but panics if an error occurs. The method allows
// neat syntax like:
//     httpReqs.With(Labels{"status":"404", "method":"POST"}).Inc()
func (m *MetricVec) With(labels Labels) Metric {
	metric, err := m.GetMetricWith(labels)
	if err != nil {
		panic(err)
	}
	return metric
}

// DeleteLabelValues removes the metric where the variable labels are the same
// as those passed in as labels (same order as the VariableLabels in Desc). It
// returns true if a metric was deleted.
//
// It is not an error if the number of label values is not the same as the
// number of VariableLabels in Desc.  However, such inconsistent label count can
// never match an actual Metric, so the method will always return false in that
// case.
//
// Note that for more than one label value, this method is prone to mistakes
// caused by an incorrect order of arguments. Consider Delete(Labels) as an
// alternative to avoid that type of mistake. For higher label numbers, the
// latter has a much more readable (albeit more verbose) syntax, but it comes
// with a performance overhead (for creating and processing the Labels map).
// See also the CounterVec example.
func (m *MetricVec) DeleteLabelValues(lvs ...string) bool {
	m.mtx.Lock()
	defer m.mtx.Unlock()

	h, err := m.hashLabelValues(lvs)
	if err != nil {
		return false
	}
	if _, has := m.children[h]; !has {
		return false
	}
	delete(m.children, h)
	return true
}

// Delete deletes the metric where the variable labels are the same as those
// passed in as labels. It returns true if a metric was deleted.
//
// It is not an error if the number and names of the Labels are inconsistent
// with those of the VariableLabels in the Desc of the MetricVec. However, such
// inconsistent Labels can never match an actual Metric, so the method will
// always return false in that case.
//
// This method is used for the same purpose as DeleteLabelValues(...string). See
// there for pros and cons of the two methods.
func (m *MetricVec) Delete(labels Labels) bool {
	m.mtx.Lock()
	defer m.mtx.Unlock()

	h, err := m.hashLabels(labels)
	if err != nil {
		return false
	}
	if _, has := m.children[h]; !has {
		return false
	}
	delete(m.children, h)
	return true
}

// Reset deletes all metrics in this vector.
func (m *MetricVec) Reset() {
	m.mtx.Lock()
	defer m.mtx.Unlock()

	for h := range m.children {
		delete(m.children, h)
	}
}

func (m *MetricVec) hashLabelValues(vals []string) (uint64, error) {
	if len(vals) != len(m.desc.variableLabels) {
		return 0, errInconsistentCardinality
	}
	m.hash.Reset()
	for _, val := range vals {
		m.buf.Reset()
		m.buf.WriteString(val)
		m.hash.Write(m.buf.Bytes())
	}
	return m.hash.Sum64(), nil
}

func (m *MetricVec) hashLabels(labels Labels) (uint64, error) {
	if len(labels) != len(m.desc.variableLabels) {
		return 0, errInconsistentCardinality
	}
	m.hash.Reset()
	for _, label := range m.desc.variableLabels {
		val, ok := labels[label]
		if !ok {
			return 0, fmt.Errorf("label name %q missing in label map", label)
		}
		m.buf.Reset()
		m.buf.WriteString(val)
		m.hash.Write(m.buf.Bytes())
	}
	return m.hash.Sum64(), nil
}

func (m *MetricVec) getOrCreateMetric(hash uint64, labelValues ...string) Metric {
	metric, ok := m.children[hash]
	if !ok {
		// Copy labelValues. Otherwise, they would be allocated even if we don't go
		// down this code path.
		copiedLabelValues := append(make([]string, 0, len(labelValues)), labelValues...)
		metric = m.newMetric(copiedLabelValues...)
		m.children[hash] = metric
	}
	return metric
}
