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

package flag

import (
	"crypto/tls"
	"reflect"
	"slices"
	"testing"
)

func TestTLSCurvePreferences(t *testing.T) {
	tests := []struct {
		flag          []string
		expected      []tls.CurveID
		expectedError bool
	}{
		{
			// Happy case: multiple curves
			flag:          []string{"secp256r1", "secp384r1", "x25519"},
			expected:      []tls.CurveID{tls.CurveP256, tls.CurveP384, tls.X25519},
			expectedError: false,
		},
		{
			// Single curve
			flag:          []string{"x25519"},
			expected:      []tls.CurveID{tls.X25519},
			expectedError: false,
		},
		{
			// Empty flag
			flag:          []string{},
			expected:      nil,
			expectedError: false,
		},
		{
			// Duplicated curve
			flag:          []string{"secp256r1", "x25519", "secp256r1"},
			expected:      []tls.CurveID{tls.CurveP256, tls.X25519, tls.CurveP256},
			expectedError: false,
		},
		{
			// Invalid curve name
			flag:          []string{"foo"},
			expected:      nil,
			expectedError: true,
		},
		{
			// All supported curves
			flag:          []string{"secp256r1", "secp384r1", "secp521r1", "x25519", "x25519mlkem768"},
			expected:      []tls.CurveID{tls.CurveP256, tls.CurveP384, tls.CurveP521, tls.X25519, tls.X25519MLKEM768},
			expectedError: false,
		},
		{
			// Case insensitive: lowercase x25519
			flag:          []string{"x25519"},
			expected:      []tls.CurveID{tls.X25519},
			expectedError: false,
		},
		{
			// Case insensitive: uppercase SECP256R1
			flag:          []string{"SECP256R1"},
			expected:      []tls.CurveID{tls.CurveP256},
			expectedError: false,
		},
		{
			// Case insensitive: mixed case
			flag:          []string{"Secp384r1", "x25519mlkem768"},
			expected:      []tls.CurveID{tls.CurveP384, tls.X25519MLKEM768},
			expectedError: false,
		},
	}

	for i, test := range tests {
		curveIDs, err := TLSCurvePreferences(test.flag)
		if !reflect.DeepEqual(curveIDs, test.expected) {
			t.Errorf("%d: expected %+v, got %+v", i, test.expected, curveIDs)
		}
		if test.expectedError && err == nil {
			t.Errorf("%d: expecting error, got %+v", i, err)
		}
		if !test.expectedError && err != nil {
			t.Errorf("%d: not expecting error, got %+v", i, err)
		}
	}
}

func TestPreferredTLSCurveNames(t *testing.T) {
	names := PreferredTLSCurveNames()
	if len(names) == 0 {
		t.Error("expected non-empty list of preferred TLS curve names")
	}
	// Verify all expected IANA names are present
	expectedNames := []string{"secp256r1", "secp384r1", "secp521r1", "x25519", "x25519mlkem768"}
	for _, expected := range expectedNames {
		if !slices.Contains(names, expected) {
			t.Errorf("expected curve name %q not found in PreferredTLSCurveNames()", expected)
		}
	}
}
