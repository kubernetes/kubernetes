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

package cgroups

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/client-go/util/exec"
)

func TestIsPodOnCgroupv2Node(t *testing.T) {
	testCases := []struct {
		name   string
		output string
		wantV2 bool
	}{
		{
			name: "cgroup v1",
		},
		{
			name:   "cgroup v2",
			output: "cgroup2 on /sys/fs/cgroup type cgroup2 (ro,nosuid,nodev,noexec,relatime)",
			wantV2: true,
		},
		{
			name:   "hybrid host mount without container cgroup v2",
			output: "cgroup2 on " + cgroupMountPath + "/unified type cgroup2 (rw,nosuid,nodev,noexec,relatime)",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			onCgroupV2, err := isPodOnCgroupv2Node(&sync.Map{}, "node-1", func() (string, string, error) {
				return tc.output, "", nil
			})
			require.NoError(t, err)
			assert.Equal(t, tc.wantV2, onCgroupV2)
		})
	}
}

func TestIsPodOnCgroupv2NodeCache(t *testing.T) {
	probeError := errors.New("connection reset by peer")
	v2Output := "cgroup2 on /sys/fs/cgroup type cgroup2 (ro,nosuid,nodev,noexec,relatime)"
	steps := []struct {
		name       string
		nodeName   string
		output     string
		stderr     string
		err        error
		wantV2     bool
		wantProbes int
	}{
		{
			name:       "failed probe with partial output is not cached",
			nodeName:   "node-a",
			output:     v2Output,
			stderr:     "probe did not complete",
			err:        probeError,
			wantProbes: 1,
		},
		{
			name:       "node-a probes again after the failure",
			nodeName:   "node-a",
			output:     v2Output,
			wantV2:     true,
			wantProbes: 2,
		},
		{
			name:       "node-a reuses the successful result",
			nodeName:   "node-a",
			wantV2:     true,
			wantProbes: 2,
		},
		{
			name:       "node-b probes independently",
			nodeName:   "node-b",
			wantProbes: 3,
		},
		{
			name:       "node-b reuses its own result",
			nodeName:   "node-b",
			output:     v2Output,
			wantProbes: 3,
		},
		{
			name:       "node-a result is unchanged by node-b",
			nodeName:   "node-a",
			wantV2:     true,
			wantProbes: 3,
		},
		{
			name:       "pod without a node name probes every time",
			output:     v2Output,
			wantV2:     true,
			wantProbes: 4,
		},
		{
			name:       "pod without a node name probes again",
			wantProbes: 5,
		},
	}
	cache := &sync.Map{}
	probeCount := 0
	for _, step := range steps {
		onCgroupV2, err := isPodOnCgroupv2Node(cache, step.nodeName, func() (string, string, error) {
			probeCount++
			return step.output, step.stderr, step.err
		})
		require.ErrorIs(t, err, step.err, step.name)
		if step.err != nil {
			require.ErrorContains(t, err, step.stderr, step.name)
		}
		assert.Equal(t, step.wantV2, onCgroupV2, step.name)
		assert.Equal(t, step.wantProbes, probeCount, "%s: probe count", step.name)
	}
}

func TestIsPodOnCgroupv2NodeWithRetry(t *testing.T) {
	v2Output := "cgroup2 on /sys/fs/cgroup type cgroup2 (ro,nosuid,nodev,noexec,relatime)"

	t.Run("retries transient exec errors and caches eventual success", func(t *testing.T) {
		cache := &sync.Map{}
		attempts := 0
		onCgroupV2, err := isPodOnCgroupv2NodeWithRetry(context.Background(), cache, "node-a", 5*time.Second, 10*time.Millisecond, func() (string, string, error) {
			attempts++
			if attempts < 3 {
				return "", "", errors.New("read: connection reset by peer")
			}
			return v2Output, "", nil
		})
		require.NoError(t, err)
		assert.True(t, onCgroupV2)
		assert.Equal(t, 3, attempts)

		cached, ok := cache.Load("node-a")
		require.True(t, ok)
		assert.Equal(t, true, cached)
	})

	t.Run("stops immediately on command exit error without caching", func(t *testing.T) {
		cache := &sync.Map{}
		attempts := 0
		_, err := isPodOnCgroupv2NodeWithRetry(context.Background(), cache, "node-b", 5*time.Second, 10*time.Millisecond, func() (string, string, error) {
			attempts++
			return "", "mount: not found", exec.CodeExitError{Err: errors.New("exit status 127"), Code: 127}
		})
		require.Error(t, err)
		assert.Equal(t, 1, attempts)
		_, ok := cache.Load("node-b")
		assert.False(t, ok)
	})

	t.Run("fails after timeout on persistent transient error without caching", func(t *testing.T) {
		cache := &sync.Map{}
		_, err := isPodOnCgroupv2NodeWithRetry(context.Background(), cache, "node-c", 50*time.Millisecond, 10*time.Millisecond, func() (string, string, error) {
			return "", "", errors.New("read: connection reset by peer")
		})
		require.Error(t, err)
		_, ok := cache.Load("node-c")
		assert.False(t, ok)
	})
}

func TestGetCPULimitCgroupExpectations(t *testing.T) {
	testCases := []struct {
		name              string
		cpuLimit          *resource.Quantity
		podOnCgroupv2Node bool
		expected          []string
	}{
		{
			name:              "rounding required, podOnCGroupv2Node=true",
			cpuLimit:          resource.NewMilliQuantity(15, resource.DecimalSI),
			podOnCgroupv2Node: true,
			expected:          []string{"1500 100000", "2000 100000"},
		},
		{
			name:              "rounding not required, podOnCGroupv2Node=true",
			cpuLimit:          resource.NewMilliQuantity(20, resource.DecimalSI),
			podOnCgroupv2Node: true,
			expected:          []string{"2000 100000"},
		},
		{
			name:              "rounding required, podOnCGroupv2Node=false",
			cpuLimit:          resource.NewMilliQuantity(15, resource.DecimalSI),
			podOnCgroupv2Node: false,
			expected:          []string{"1500", "2000"},
		},
		{
			name:              "rounding not required, podOnCGroupv2Node=false",
			cpuLimit:          resource.NewMilliQuantity(20, resource.DecimalSI),
			podOnCgroupv2Node: false,
			expected:          []string{"2000"},
		},
		{
			name:              "minimum quota, podOnCGroupv2Node=true",
			cpuLimit:          resource.NewMilliQuantity(5, resource.DecimalSI),
			podOnCgroupv2Node: true,
			expected:          []string{"1000 100000", "1000 100000"},
		},
		{
			name:              "cpuQuota=0, podOnCGroupv2Node=true",
			cpuLimit:          resource.NewMilliQuantity(0, resource.DecimalSI),
			podOnCgroupv2Node: true,
			expected:          []string{"max 100000"},
		},
		{
			name:              "cpuQuota=0, podOnCGroupv2Node=false",
			cpuLimit:          resource.NewMilliQuantity(0, resource.DecimalSI),
			podOnCgroupv2Node: false,
			expected:          []string{"-1"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := getCPULimitCgroupExpectations(tc.cpuLimit, tc.podOnCgroupv2Node)
			assert.Equal(t, tc.expected, actual)
		})
	}
}
