// Copyright 2014 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package appengine

import (
	"testing"

	"golang.org/x/net/context"
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
		_, err := Namespace(context.Background(), tc.namespace)
		if err == nil && !tc.ok {
			t.Errorf("Namespace %q should be rejected, but wasn't", tc.namespace)
		} else if err != nil && tc.ok {
			t.Errorf("Namespace %q should be accepted, but wasn't", tc.namespace)
		}
	}
}
