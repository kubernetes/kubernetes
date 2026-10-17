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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
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

func TestSystemPartitionConfigHasPod(t *testing.T) {
	podIn := func(namespace string) *v1.Pod {
		return &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: namespace}}
	}

	testCases := []struct {
		name   string
		config *SystemPartitionConfig
		pod    *v1.Pod
		want   bool
	}{
		{
			name:   "nil namespace set",
			config: &SystemPartitionConfig{},
			pod:    podIn("kube-system"),
			want:   false,
		},
		{
			name:   "empty namespace set",
			config: &SystemPartitionConfig{Namespaces: sets.New[string]()},
			pod:    podIn("kube-system"),
			want:   false,
		},
		{
			name:   "listed namespace",
			config: &SystemPartitionConfig{Namespaces: sets.New("kube-system")},
			pod:    podIn("kube-system"),
			want:   true,
		},
		{
			name:   "unlisted namespace",
			config: &SystemPartitionConfig{Namespaces: sets.New("kube-system")},
			pod:    podIn("default"),
			want:   false,
		},
		{
			name:   "one of several listed namespaces",
			config: &SystemPartitionConfig{Namespaces: sets.New("kube-system", "monitoring")},
			pod:    podIn("monitoring"),
			want:   true,
		},
		{
			name:   "no partition config",
			config: nil,
			pod:    podIn("kube-system"),
			want:   false,
		},
		{
			name:   "no pod",
			config: &SystemPartitionConfig{Namespaces: sets.New("kube-system")},
			pod:    nil,
			want:   false,
		},
		{
			name:   "no partition config and pod",
			config: nil,
			pod:    nil,
			want:   false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, tc.config.HasPod(tc.pod))
		})
	}
}
