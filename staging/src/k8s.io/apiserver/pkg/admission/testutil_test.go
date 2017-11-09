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
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	ptype "github.com/prometheus/client_model/go"
)

// FakeHandler provide a mock implement both MutationInterface and ValidationInterface that tracks which
// methods have been called and always returns an error if admit is false.
type FakeHandler struct {
	*Handler
	admit          bool
	admitCalled    bool
	validateCalled bool
}

func (h *FakeHandler) Admit(a Attributes) (err error) {
	h.admitCalled = true
	if h.admit {
		return nil
	}
	return fmt.Errorf("Don't admit")
}

func (h *FakeHandler) Validate(a Attributes) (err error) {
	h.validateCalled = true
	if h.admit {
		return nil
	}
	return fmt.Errorf("Don't admit")
}

func makeHandler(admit bool, ops ...Operation) *FakeHandler {
	return &FakeHandler{
		admit:   admit,
		Handler: NewHandler(ops...),
	}
}

func makeNamedHandler(name string, admit bool, ops ...Operation) NamedHandler {
	return &pluginHandler{
		i: &FakeHandler{
			admit:   admit,
			Handler: NewHandler(ops...),
		},
		name: name,
	}
}

// FakeValidatingHandler provide a mock of ValidationInterface that tracks which
// methods have been called and always returns an error if validate is false.
type FakeValidatingHandler struct {
	*Handler
	validate, validateCalled bool
}

func (h *FakeValidatingHandler) Validate(a Attributes) (err error) {
	h.validateCalled = true
	if h.validate {
		return nil
	}
	return fmt.Errorf("Don't validate")
}

func makeValidatingHandler(validate bool, ops ...Operation) *FakeValidatingHandler {
	return &FakeValidatingHandler{
		validate: validate,
		Handler:  NewHandler(ops...),
	}
}

func makeValidatingNamedHandler(name string, validate bool, ops ...Operation) NamedHandler {
	return &pluginHandler{
		i: &FakeValidatingHandler{
			validate: validate,
			Handler:  NewHandler(ops...),
		},
		name: name,
	}
}

func labelsMatch(metric *ptype.Metric, labelFilter map[string]string) bool {
	for _, lp := range metric.GetLabel() {
		if value, ok := labelFilter[lp.GetName()]; ok && lp.GetValue() != value {
			return false
		}
	}
	return true
}

// expectFindMetric find a metric with the given name nad labels or reports a fatal test error.
func expectFindMetric(t *testing.T, name string, expectedLabels map[string]string) *ptype.Metric {
	metrics, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("Failed to gather metrics: %s", err)
	}

	for _, mf := range metrics {
		if mf.GetName() == name {
			for _, metric := range mf.GetMetric() {
				if labelsMatch(metric, expectedLabels) {
					gotLabelCount := len(metric.GetLabel())
					wantLabelCount := len(expectedLabels)
					if wantLabelCount != gotLabelCount {
						t.Errorf("Got metric with %d labels, but wanted %d labels. Wanted %#+v for %s",
							gotLabelCount, wantLabelCount, expectedLabels, metric.String())
					}
					return metric
				}
			}
		}
	}
	t.Fatalf("No metric found with name %s and labels %#+v", name, expectedLabels)
	return nil
}

// expectHistogramCountTotal ensures that the sum of counts of metrics matching the labelFilter is as
// expected.
func expectHistogramCountTotal(t *testing.T, name string, labelFilter map[string]string, wantCount int) {
	metrics, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("Failed to gather metrics: %s", err)
	}

	counterSum := 0
	for _, mf := range metrics {
		if mf.GetName() != name {
			continue // Ignore other metrics.
		}
		for _, metric := range mf.GetMetric() {
			if !labelsMatch(metric, labelFilter) {
				continue
			}
			counterSum += int(metric.GetHistogram().GetSampleCount())
		}
	}
	if wantCount != counterSum {
		t.Errorf("Wanted count %d, got %d for metric %s with labels %#+v", wantCount, counterSum, name, labelFilter)
		for _, mf := range metrics {
			if mf.GetName() == name {
				for _, metric := range mf.GetMetric() {
					t.Logf("\tnear match: %s", metric.String())
				}
			}
		}
	}
}
