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

package admission

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	ptype "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type FakeHandler struct {
	*Handler
	name                     string
	admit, admitCalled       bool
	validate, validateCalled bool
}

func (h *FakeHandler) GetName() string {
	return h.name
}

func (h *FakeHandler) Admit(a Attributes) (err error) {
	h.admitCalled = true
	if h.admit {
		return nil
	}
	return fmt.Errorf("Don't admit")
}

func (h *FakeHandler) Validate(a Attributes) (err error) {
	h.admitCalled = true
	if h.admit {
		return nil
	}
	return fmt.Errorf("Don't admit")
}

// makeHandler creates a mock handler for testing purposes.
func makeHandler(name string, admit bool, ops ...Operation) *FakeHandler {
	return &FakeHandler{
		name:    name,
		admit:   admit,
		Handler: NewHandler(ops...),
	}
}

type metricLabels struct {
	operation   string
	group       string
	version     string
	resource    string
	subresource string
	name        string
	tpe         string
	isSystemNs  bool
}

// matches checks if the reciever matches the pattern. Empty strings in the pattern are treated as wildcards.
func (l metricLabels) matches(pattern metricLabels) bool {
	return matches(l.operation, pattern.operation) &&
		matches(l.group, pattern.group) &&
		matches(l.version, pattern.version) &&
		matches(l.resource, pattern.resource) &&
		matches(l.subresource, pattern.subresource) &&
		matches(l.tpe, pattern.tpe) &&
		l.isSystemNs == pattern.isSystemNs
}

// matches checks if a string matches a "pattern" string, where an empty pattern string is treated as a wildcard.
func matches(s string, pattern string) bool {
	return pattern == "" || s == pattern
}

// readLabels marshalls the labels from a prometheus metric type to a simple struct, producing test errors if
// if any unrecognized labels are encountered.
func readLabels(t *testing.T, metric *ptype.Metric) metricLabels {
	l := metricLabels{}
	for _, lp := range metric.GetLabel() {
		val := lp.GetValue()
		switch lp.GetName() {
		case "operation":
			l.operation = val
		case "group":
			l.group = val
		case "version":
			l.version = val
		case "resource":
			l.resource = val
		case "subresource":
			l.subresource = val
		case "name":
			l.name = val
		case "type":
			l.tpe = val
		case "is_system_ns":
			ns, err := strconv.ParseBool(lp.GetValue())
			if err != nil {
				t.Errorf("Expected boole for is_system_ns label value, got %s", lp.GetValue())
			} else {
				l.isSystemNs = ns
			}
		default:
			t.Errorf("Unexpected metric label %s", lp.GetName())
		}
	}
	return l
}

// expectCounterMetric ensures that exactly one counter metric with the given name and patternLabels exists and has
// the provided count.
func expectCountMetric(t *testing.T, name string, patternLabels metricLabels, wantCount int64) {
	metrics, err := prometheus.DefaultGatherer.Gather()
	require.NoError(t, err)

	count := 0
	for _, mf := range metrics {
		if mf.GetName() != name {
			continue // Ignore other metrics.
		}
		for _, metric := range mf.GetMetric() {
			gotLabels := readLabels(t, metric)
			if !gotLabels.matches(patternLabels) {
				continue
			}
			count += 1
			assert.EqualValues(t, wantCount, metric.GetCounter().GetValue())
		}
	}
	if count != 1 {
		t.Errorf("Want 1 metric with name %s, got %d", name, count)
	}
}
