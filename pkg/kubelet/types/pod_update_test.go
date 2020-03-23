/*
Copyright 2014 The Kubernetes Authors.

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

package types

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGetValidatedSources(t *testing.T) {
	// Empty.
	sources, err := GetValidatedSources([]string{""})
	require.NoError(t, err)
	require.Len(t, sources, 0)

	// Success.
	sources, err = GetValidatedSources([]string{FileSource, ApiserverSource})
	require.NoError(t, err)
	require.Len(t, sources, 2)

	// All.
	sources, err = GetValidatedSources([]string{AllSource})
	require.NoError(t, err)
	require.Len(t, sources, 3)

	// Unknown source.
	_, err = GetValidatedSources([]string{"taco"})
	require.Error(t, err)
}

func TestGetPodSource(t *testing.T) {
	cases := []struct {
		pod         v1.Pod
		expected    string
		errExpected bool
	}{
		{
			pod:         v1.Pod{},
			expected:    "",
			errExpected: true,
		},
		{
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"kubernetes.io/config.source": "host-ipc-sources",
					},
				},
			},
			expected:    "host-ipc-sources",
			errExpected: false,
		},
	}
	for i, data := range cases {
		source, err := GetPodSource(&data.pod)
		if data.errExpected {
			assert.Error(t, err)
		} else {
			assert.NoError(t, err)
		}
		assert.Equal(t, data.expected, source, "test[%d]", i)
		t.Logf("Test case [%d]", i)
	}
}

func TestString(t *testing.T) {
	cases := []struct {
		sp       SyncPodType
		expected string
	}{
		{
			sp:       SyncPodCreate,
			expected: "create",
		},
		{
			sp:       SyncPodUpdate,
			expected: "update",
		},
		{
			sp:       SyncPodSync,
			expected: "sync",
		},
		{
			sp:       SyncPodKill,
			expected: "kill",
		},
		{
			sp:       50,
			expected: "unknown",
		},
	}
	for i, data := range cases {
		syncPodString := data.sp.String()
		assert.Equal(t, data.expected, syncPodString, "test[%d]", i)
		t.Logf("Test case [%d]", i)
	}
}

func TestIsCriticalPodBasedOnPriority(t *testing.T) {
	tests := []struct {
		priority    int32
		description string
		expected    bool
	}{
		{
			priority:    int32(2000000001),
			description: "A system critical pod",
			expected:    true,
		},
		{
			priority:    int32(1000000000),
			description: "A non system critical pod",
			expected:    false,
		},
	}
	for _, test := range tests {
		actual := IsCriticalPodBasedOnPriority(test.priority)
		if actual != test.expected {
			t.Errorf("IsCriticalPodBased on priority should have returned %v for test %v but got %v", test.expected, test.description, actual)
		}
	}
}
