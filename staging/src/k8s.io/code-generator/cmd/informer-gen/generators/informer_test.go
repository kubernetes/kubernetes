/*
Copyright 2026 The Kubernetes Authors.

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

package generators

import (
	"testing"
)

func TestPluralizeName(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		// Regular plurals (add 's')
		{
			name:     "simple singular",
			input:    "Pod",
			expected: "pods",
		},
		{
			name:     "simple singular uppercase",
			input:    "Node",
			expected: "nodes",
		},
		// Words ending in 'y' preceded by consonant → strip 'y', add 'ies'
		{
			name:     "y-ending with consonant",
			input:    "Policy",
			expected: "policies",
		},
		{
			name:     "affinity-like",
			input:    "BlockAffinity",
			expected: "blockaffinities",
		},
		{
			name:     "NetworkPolicy",
			input:    "GlobalNetworkPolicy",
			expected: "globalnetworkpolicies",
		},
		// Words ending in 's' → add 'es'
		{
			name:     "s-ending",
			input:    "Class",
			expected: "classes",
		},
		{
			name:     "s-ending componentstatus",
			input:    "ComponentStatus",
			expected: "componentstatuses",
		},
		// Words ending in 'ch' → add 'es'
		{
			name:     "ch-ending",
			input:    "Watch",
			expected: "watches",
		},
		// Words ending in 'sh' → add 'es'
		{
			name:     "sh-ending",
			input:    "Trash",
			expected: "trashes",
		},
		// Words ending in 'x' → add 'es'
		{
			name:     "x-ending",
			input:    "Index",
			expected: "indexes",
		},
		// Words ending in 'z' → add 'es'
		{
			name:     "z-ending",
			input:    "Quiz",
			expected: "quizzes",
		},
		// Words ending in 'y' preceded by vowel → add 's'
		{
			name:     "y-ending with vowel",
			input:    "Array",
			expected: "arrays",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := pluralizeName(tt.input)
			if result != tt.expected {
				t.Errorf("pluralizeName(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestIsVowel(t *testing.T) {
	tests := []struct {
		r        rune
		expected bool
	}{
		{'a', true},
		{'e', true},
		{'i', true},
		{'o', true},
		{'u', true},
		{'b', false},
		{'c', false},
		{'y', false},
	}

	for _, tt := range tests {
		t.Run(string(tt.r), func(t *testing.T) {
			result := isVowel(tt.r)
			if result != tt.expected {
				t.Errorf("isVowel(%q) = %v, want %v", tt.r, result, tt.expected)
			}
		})
	}
}
