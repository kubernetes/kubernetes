//go:build linux

/*
Copyright The Kubernetes Authors.

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

package cm

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestParsePercentage(t *testing.T) {
	testCases := []struct {
		name      string
		input     string
		expectErr bool
		expected  int64
	}{
		{
			name:      "valid zero percentage",
			input:     "0%",
			expected:  0,
			expectErr: false,
		},
		{
			name:      "valid mid percentage",
			input:     "50%",
			expected:  50,
			expectErr: false,
		},
		{
			name:      "valid max percentage",
			input:     "100%",
			expected:  100,
			expectErr: false,
		},
		{
			name:      "missing percentage sign",
			input:     "50",
			expectErr: true,
		},
		{
			name:      "negative percentage",
			input:     "-1%",
			expectErr: true,
		},
		{
			name:      "percentage greater than 100",
			input:     "101%",
			expectErr: true,
		},
		{
			name:      "non-numeric percentage",
			input:     "abc%",
			expectErr: true,
		},
		{
			name:      "empty string",
			input:     "",
			expectErr: true,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			result, err := parsePercentage(testCase.input)

			if testCase.expectErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.Equal(t, testCase.expected, result)
		})
	}
}
