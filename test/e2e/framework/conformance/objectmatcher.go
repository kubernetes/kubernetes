/*
Copyright 2025 The Kubernetes Authors.

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

package architecture

import (
	"fmt"
	"strings"

	"github.com/onsi/gomega/format"
	gtypes "github.com/onsi/gomega/types"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
)

// matchObjectList is a custom matcher for the result of a generic lister List result
// ([]runtime.Object where each entry is *unstructured.Unstructed). The list is
// expected to have exactly one element or none.
type matchObjectList struct {
	expectedObject *unstructured.Unstructured
	verifyContent  bool
	lastDiff       string
}

var _ gtypes.GomegaMatcher = &matchObjectList{}

func (m *matchObjectList) Match(actual any) (bool, error) {
	// Reset state.
	m.lastDiff = ""

	actualObjects, ok := actual.([]runtime.Object)
	if !ok {
		return false, fmt.Errorf("must be passed a []runtime.Object, got %T", actual)
	}

	if m.expectedObject == nil {
		// Must be empty,
		return len(actualObjects) == 0, nil
	}

	// Must have exactly the expected object.
	if len(actualObjects) != 1 {
		return false, nil
	}
	actualObject, ok := actualObjects[0].(*unstructured.Unstructured)
	if !ok {
		// Shouldn't happen.
		return false, fmt.Errorf("expected *unstructured.Unstructured, got %T", actualObjects[0])
	}

	if m.verifyContent {
		// Remember diff for failure message.
		m.lastDiff = compareObjects(m.expectedObject, actualObject)
		if m.lastDiff != "" {
			return false, nil
		}
		return true, nil
	}
	return m.expectedObject.GetName() == actualObject.GetName() && m.expectedObject.GetNamespace() == actualObject.GetNamespace(), nil
}

func (m *matchObjectList) FailureMessage(actual any) string {
	return m.message(actual, "to")
}

func (m *matchObjectList) NegatedFailureMessage(actual any) string {
	return m.message(actual, "not to")
}

func (m *matchObjectList) message(actual any, to string) string {
	// Gomega renders []runtime.Object as nested maps.
	// YAML is more readable.
	var buffer strings.Builder
	buffer.WriteString("Expected\n")
	if actualObjects, ok := actual.([]runtime.Object); ok {
		buffer.WriteString(fmt.Sprintf("   %T len:%d:\n", actualObjects, len(actualObjects)))
		for _, object := range actualObjects {
			buffer.WriteString("      ---\n")
			if o, ok := object.(*unstructured.Unstructured); ok {
				buffer.WriteString(format.Object(o, 2))
			} else {
				buffer.WriteString(format.Object(object, 2))
			}
		}
	} else {
		buffer.WriteString(format.Object(actual, 1))
	}
	buffer.WriteString(fmt.Sprintf("\n%s contain exactly the following element:\n", to))
	if m.verifyContent {
		buffer.WriteString(format.Object(m.expectedObject, 1))
	} else {
		buffer.WriteString("      " + klog.KObj(m.expectedObject).String())
	}
	if m.lastDiff != "" {
		buffer.WriteString("\nDiff of checked fields (- expected, + actual):\n")
		buffer.WriteString(m.lastDiff)
	}
	return buffer.String()
}
