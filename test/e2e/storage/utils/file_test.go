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

package utils

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestShortenFileName(t *testing.T) {
	tests := []struct {
		name     string
		filename string
		expected string
	}{
		{
			name:     "Shorter than max length",
			filename: "short file name",
			expected: "short file name",
		},
		{
			name:     "Longer than max length, truncated",
			filename: "a very long string that has exactly 256 characters a very long string that has exactly 256 characters a very long string that has exactly 256 characters a very long string that has exactly 256 characters a very long string that has exactly 256 characters..",
			expected: "a very long string that has exactly 256 characters a very long string that has exactly 256 characters a very long string that has exactly 256 characters a very long string that has exactly 256 characters a very long string that has exactly 256 ch-ad31f675",
		},
		{
			name:     "Exactly max length, not truncated",
			filename: "a very long string that has exactly 255 characters a very long string that has exactly 255 characters a very long string that has exactly 255 characters a very long string that has exactly 255 characters a very long string that has exactly 255 characters.",
			expected: "a very long string that has exactly 255 characters a very long string that has exactly 255 characters a very long string that has exactly 255 characters a very long string that has exactly 255 characters a very long string that has exactly 255 characters.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ShortenFileName(tt.filename)
			assert.Equal(t, tt.expected, result)
			assert.LessOrEqual(t, len(result), maxFileNameLength)
		})
	}
}
