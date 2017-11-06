/*
Copyright 2015 The Kubernetes Authors.

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
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

type FakeHandler struct {
	*Handler
	name                     string
	admit, admitCalled       bool
	validate, validateCalled bool
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

func makeHandler(name string, admit bool, ops ...Operation) *FakeHandler {
	return &FakeHandler{
		name:    name,
		admit:   admit,
		Handler: NewHandler(ops...),
	}
}

func makeChain(handlers ...*FakeHandler) chainAdmissionHandler {
	chain := chainAdmissionHandler{}
	for _, fh := range handlers {
		chain = chain.Append(fh.name, fh)
	}
	return chain
}

func TestAdmit(t *testing.T) {
	sysns := "kube-system"
	otherns := "default"
	tests := []struct {
		name      string
		ns        string
		operation Operation
		chain     []*FakeHandler
		accept    bool
		reject    string
		calls     map[string]bool
	}{
		{
			name:      "all accept",
			ns:        sysns,
			operation: Create,
			chain: []*FakeHandler{
				makeHandler("a", true, Update, Delete, Create),
				makeHandler("b", true, Delete, Create),
				makeHandler("c", true, Create),
			},
			calls:  map[string]bool{"a": true, "b": true, "c": true},
			accept: true,
		},
		{
			name:      "ignore handler",
			ns:        otherns,
			operation: Create,
			chain: []*FakeHandler{
				makeHandler("a", true, Update, Delete, Create),
				makeHandler("b", false, Delete),
				makeHandler("c", true, Create),
			},
			calls:  map[string]bool{"a": true, "c": true},
			accept: true,
		},
		{
			name:      "ignore all",
			ns:        sysns,
			operation: Connect,
			chain: []*FakeHandler{
				makeHandler("a", true, Update, Delete, Create),
				makeHandler("b", false, Delete),
				makeHandler("c", true, Create),
			},
			calls:  map[string]bool{},
			accept: true,
		},
		{
			name:      "reject one",
			ns:        otherns,
			operation: Delete,
			chain: []*FakeHandler{
				makeHandler("a", true, Update, Delete, Create),
				makeHandler("b", false, Delete),
				makeHandler("c", true, Create),
			},
			calls:  map[string]bool{"a": true, "b": true},
			accept: false,
			reject: "b",
		},
	}
	for _, test := range tests {
		t.Logf("testcase = %s", test.name)
		chain := makeChain(test.chain...)
		resetMetrics()
		err := chain.Admit(NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, test.ns, "", schema.GroupVersionResource{}, "", test.operation, nil))
		accepted := (err == nil)
		if accepted != test.accept {
			t.Errorf("unexpected result of admit call: %v", accepted)
		}
		for _, fake := range test.chain {
			_, shouldBeCalled := test.calls[fake.name]
			if shouldBeCalled != fake.admitCalled {
				t.Errorf("handler %s not called as expected: %v", fake.name, fake.admitCalled)
				continue
			}
		}
		expectMetrics(t, test.reject, test.ns == sysns)
	}
}

func TestHandles(t *testing.T) {
	chain := makeChain(
		makeHandler("a", true, Update, Delete, Create),
		makeHandler("b", true, Delete, Create),
		makeHandler("c", true, Create),
	)

	tests := []struct {
		name      string
		operation Operation
		chain     chainAdmissionHandler
		expected  bool
	}{
		{
			name:      "all handle",
			operation: Create,
			expected:  true,
		},
		{
			name:      "none handle",
			operation: Connect,
			expected:  false,
		},
		{
			name:      "some handle",
			operation: Delete,
			expected:  true,
		},
	}
	for _, test := range tests {
		handles := chain.Handles(test.operation)
		if handles != test.expected {
			t.Errorf("Unexpected handles result. Expected: %v. Actual: %v", test.expected, handles)
		}
	}
}

// Expect these metrics to be incremented after a single call to Admit.
func expectMetrics(t *testing.T, reject string, systemNs bool) {
	metrics, err := prometheus.DefaultGatherer.Gather()
	require.NoError(t, err)

	for _, mf := range metrics {
		if !strings.HasPrefix(mf.GetName(), "apiserver_admission_") {
			continue // Ignore other metrics.
		}
		for _, metric := range mf.GetMetric() {
			for _, lp := range metric.GetLabel() {
				switch lp.GetName() {
				case "is_system_ns":
					assert.Equal(t, strconv.FormatBool(systemNs), lp.GetValue(), "metric=%s", mf.GetName())
				case "plugin":
					assert.Equal(t, reject, lp.GetValue(), "metric=%s", mf.GetName())
				default:
					t.Errorf("Unexpected metric label %s on %s", lp.GetName(), mf.GetName())
				}
			}
			switch mf.GetName() {
			case "apiserver_admission_handle_total":
				assert.EqualValues(t, 1, metric.GetCounter().GetValue())
			case "apiserver_admission_reject_total":
				if reject == "" {
					t.Errorf("Unexpected reject")
				}
				assert.EqualValues(t, 1, metric.GetCounter().GetValue())
			default:
				t.Errorf("Unexpected metric: %s", mf.GetName())
				continue
			}
		}
	}
}

func resetMetrics() {
	handleCounter.Reset()
	rejectCounter.Reset()
}
