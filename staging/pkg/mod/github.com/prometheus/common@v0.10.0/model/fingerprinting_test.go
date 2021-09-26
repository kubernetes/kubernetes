// Copyright 2019 The Prometheus Authors
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
	"sort"
	"testing"
)

func TestFingerprintFromString(t *testing.T) {
	fs := "4294967295"

	f, err := FingerprintFromString(fs)

	if err != nil {
		t.Errorf("unexpected error while getting Fingerprint from string: %s", err.Error())
	}

	expected := Fingerprint(285960729237)

	if expected != f {
		t.Errorf("expected to get %d, but got %d instead", f, expected)
	}

	f, err = ParseFingerprint(fs)

	if err != nil {
		t.Errorf("unexpected error while getting Fingerprint from string: %s", err.Error())
	}

	if expected != f {
		t.Errorf("expected to get %d, but got %d instead", f, expected)
	}
}

func TestFingerprintsSort(t *testing.T) {
	fingerPrints := Fingerprints{
		14695981039346656037,
		285960729237,
		0,
		4294967295,
		285960729237,
		18446744073709551615,
	}

	sort.Sort(fingerPrints)

	expected := Fingerprints{
		0,
		4294967295,
		285960729237,
		285960729237,
		14695981039346656037,
		18446744073709551615,
	}

	for i, f := range fingerPrints {
		if f != expected[i] {
			t.Errorf("expected Fingerprint %d, but got %d for index %d", expected[i], f, i)
		}
	}
}

func TestFingerprintSet(t *testing.T) {
	// Testing with two sets of unequal length.
	f := FingerprintSet{
		14695981039346656037: struct{}{},
		0:                    struct{}{},
		4294967295:           struct{}{},
		285960729237:         struct{}{},
		18446744073709551615: struct{}{},
	}

	f2 := FingerprintSet{
		285960729237: struct{}{},
	}

	if f.Equal(f2) {
		t.Errorf("expected two FingerPrintSets of unequal length to be unequal")
	}

	// Testing with two unequal sets of equal length.
	f = FingerprintSet{
		14695981039346656037: struct{}{},
		0:                    struct{}{},
		4294967295:           struct{}{},
	}

	f2 = FingerprintSet{
		14695981039346656037: struct{}{},
		0:                    struct{}{},
		285960729237:         struct{}{},
	}

	if f.Equal(f2) {
		t.Errorf("expected two FingerPrintSets of unequal content to be unequal")
	}

	// Testing with equal sets of equal length.
	f = FingerprintSet{
		14695981039346656037: struct{}{},
		0:                    struct{}{},
		4294967295:           struct{}{},
	}

	f2 = FingerprintSet{
		14695981039346656037: struct{}{},
		0:                    struct{}{},
		4294967295:           struct{}{},
	}

	if !f.Equal(f2) {
		t.Errorf("expected two FingerPrintSets of equal content to be equal")
	}
}

func TestFingerprintIntersection(t *testing.T) {
	scenarios := []struct {
		name     string
		input1   FingerprintSet
		input2   FingerprintSet
		expected FingerprintSet
	}{
		{
			name:     "two empty sets",
			input1:   FingerprintSet{},
			input2:   FingerprintSet{},
			expected: FingerprintSet{},
		},
		{
			name: "one empty set",
			input1: FingerprintSet{
				0: struct{}{},
			},
			input2:   FingerprintSet{},
			expected: FingerprintSet{},
		},
		{
			name: "two non-empty unequal sets",
			input1: FingerprintSet{
				14695981039346656037: struct{}{},
				0:                    struct{}{},
				4294967295:           struct{}{},
			},

			input2: FingerprintSet{
				14695981039346656037: struct{}{},
				0:                    struct{}{},
				4294967295:           struct{}{},
			},
			expected: FingerprintSet{
				14695981039346656037: struct{}{},
				0:                    struct{}{},
				4294967295:           struct{}{},
			},
		},
		{
			name: "two non-empty equal sets",
			input1: FingerprintSet{
				14695981039346656037: struct{}{},
				0:                    struct{}{},
				285960729237:         struct{}{},
			},

			input2: FingerprintSet{
				14695981039346656037: struct{}{},
				0:                    struct{}{},
				4294967295:           struct{}{},
			},
			expected: FingerprintSet{
				14695981039346656037: struct{}{},
				0:                    struct{}{},
			},
		},
		{
			name: "two non-empty equal sets of unequal length",
			input1: FingerprintSet{
				14695981039346656037: struct{}{},
				0:                    struct{}{},
				285960729237:         struct{}{},
			},

			input2: FingerprintSet{
				14695981039346656037: struct{}{},
				0:                    struct{}{},
			},
			expected: FingerprintSet{
				14695981039346656037: struct{}{},
				0:                    struct{}{},
			},
		},
	}

	for _, scenario := range scenarios {
		s1 := scenario.input1
		s2 := scenario.input2
		actual := s1.Intersection(s2)

		if !actual.Equal(scenario.expected) {
			t.Errorf("expected %v to be equal to %v", actual, scenario.expected)
		}
	}
}
