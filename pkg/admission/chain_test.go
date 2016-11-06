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
	"testing"

	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/util/errors"
)

type FakeHandler struct {
	*Handler
	name        string
	admit       bool
	admitCalled bool
	warning     Warning
}

func (h *FakeHandler) Admit(a Attributes) (warn Warning, err error) {
	h.admitCalled = true
	if h.admit {
		return h.warning, nil
	}
	return h.warning, fmt.Errorf("Don't admit")
}

func makeHandler(name string, admit bool, ops ...Operation) Interface {
	return &FakeHandler{
		name:    name,
		admit:   admit,
		Handler: NewHandler(ops...),
	}
}

func makeHandlerWithWarning(name string, admit bool, warn Warning, ops ...Operation) Interface {
	return &FakeHandler{
		name:    name,
		admit:   admit,
		Handler: NewHandler(ops...),
		warning: warn,
	}
}

func containsWarning(warning Warning, warnings []error) bool {
	for ix := range warnings {
		if warning.Error() == warnings[ix].Error() {
			return true
		}
	}
	return false
}

func TestAdmit(t *testing.T) {
	tests := []struct {
		name      string
		operation Operation
		chain     chainAdmissionHandler
		accept    bool
		warnings  bool
		calls     map[string]bool
	}{
		{
			name:      "all accept",
			operation: Create,
			chain: []Interface{
				makeHandler("a", true, Update, Delete, Create),
				makeHandler("b", true, Delete, Create),
				makeHandler("c", true, Create),
			},
			calls:  map[string]bool{"a": true, "b": true, "c": true},
			accept: true,
		},
		{
			name:      "ignore handler",
			operation: Create,
			chain: []Interface{
				makeHandler("a", true, Update, Delete, Create),
				makeHandler("b", false, Delete),
				makeHandler("c", true, Create),
			},
			calls:  map[string]bool{"a": true, "c": true},
			accept: true,
		},
		{
			name:      "ignore all",
			operation: Connect,
			chain: []Interface{
				makeHandler("a", true, Update, Delete, Create),
				makeHandler("b", false, Delete),
				makeHandler("c", true, Create),
			},
			calls:  map[string]bool{},
			accept: true,
		},
		{
			name:      "reject one",
			operation: Delete,
			chain: []Interface{
				makeHandler("a", true, Update, Delete, Create),
				makeHandler("b", false, Delete),
				makeHandler("c", true, Create),
			},
			calls:  map[string]bool{"a": true, "b": true},
			accept: false,
		},
		{
			name:      "warnings one",
			operation: Create,
			chain: []Interface{
				makeHandlerWithWarning("a", true, Warning(fmt.Errorf("fake warning")), Update, Delete, Create),
				makeHandler("b", true, Create),
				makeHandlerWithWarning("c", true, Warning(fmt.Errorf("fake warning 2")), Create),
			},
			calls:    map[string]bool{"a": true, "b": true, "c": true},
			accept:   true,
			warnings: true,
		},
	}
	for _, test := range tests {
		warn, err := test.chain.Admit(NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, "", "", schema.GroupVersionResource{}, "", test.operation, nil))
		accepted := (err == nil)
		if accepted != test.accept {
			t.Errorf("%s: unexpected result of admit call: %v\n", test.name, accepted)
		}
		if test.warnings && warn == nil {
			t.Errorf("unexpected non-warning")
		}
		var warnings []error
		if warn != nil {
			aggregate, ok := warn.(errors.Aggregate)
			if !ok {
				t.Errorf("unexpected warning: %v", warn)
			} else {
				warnings = aggregate.Errors()
			}
		}

		for _, h := range test.chain {
			fake := h.(*FakeHandler)
			_, shouldBeCalled := test.calls[fake.name]
			if shouldBeCalled != fake.admitCalled {
				t.Errorf("%s: handler %s not called as expected: %v", test.name, fake.name, fake.admitCalled)
				continue
			}
			if fake.warning != nil && !containsWarning(fake.warning, warnings) {
				t.Errorf("%s: failed to find expected warning: %v (%v)", test.name, fake.warning, warnings)
			}
		}
	}
}

func TestHandles(t *testing.T) {
	chain := chainAdmissionHandler{
		makeHandler("a", true, Update, Delete, Create),
		makeHandler("b", true, Delete, Create),
		makeHandler("c", true, Create),
	}

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
