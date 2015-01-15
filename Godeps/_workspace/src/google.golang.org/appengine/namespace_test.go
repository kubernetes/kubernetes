package appengine

import (
	"testing"
)

func TestNamespaceValidity(t *testing.T) {
	testCases := []struct {
		namespace string
		ok        bool
	}{
		// data from Python's namespace_manager_test.py
		{"", true},
		{"__a.namespace.123__", true},
		{"-_A....NAMESPACE-_", true},
		{"-", true},
		{".", true},
		{".-", true},

		{"?", false},
		{"+", false},
		{"!", false},
		{" ", false},
	}
	for _, tc := range testCases {
		_, err := Namespace(nil, tc.namespace)
		if err == nil && !tc.ok {
			t.Errorf("Namespace %q should be rejected, but wasn't", tc.namespace)
		} else if err != nil && tc.ok {
			t.Errorf("Namespace %q should be accepted, but wasn't", tc.namespace)
		}
	}
}
