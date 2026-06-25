//go:build linux

/*
Copyright 2016 The Kubernetes Authors.

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
	"path"
	"reflect"
	"testing"

	libcontainercgroups "github.com/opencontainers/cgroups"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/cpuset"
	"k8s.io/utils/ptr"
)

// TestNewCgroupName tests confirms that #68416 is fixed
func TestNewCgroupName(t *testing.T) {
	a := ParseCgroupfsToCgroupName("/a/")
	ab := NewCgroupName(a, "b")

	expectedAB := CgroupName([]string{"a", "", "b"})
	if !reflect.DeepEqual(ab, expectedAB) {
		t.Errorf("Expected %d%+v; got %d%+v", len(expectedAB), expectedAB, len(ab), ab)
	}

	abc := NewCgroupName(ab, "c")

	expectedABC := CgroupName([]string{"a", "", "b", "c"})
	if !reflect.DeepEqual(abc, expectedABC) {
		t.Errorf("Expected %d%+v; got %d%+v", len(expectedABC), expectedABC, len(abc), abc)
	}

	_ = NewCgroupName(ab, "d")

	if !reflect.DeepEqual(abc, expectedABC) {
		t.Errorf("Expected %d%+v; got %d%+v", len(expectedABC), expectedABC, len(abc), abc)
	}
}

func TestCgroupNameToSystemdBasename(t *testing.T) {
	testCases := []struct {
		input    CgroupName
		expected string
	}{
		{
			input:    RootCgroupName,
			expected: "/",
		},
		{
			input:    NewCgroupName(RootCgroupName, "system"),
			expected: "system.slice",
		},
		{
			input:    NewCgroupName(RootCgroupName, "system", "Burstable"),
			expected: "system-Burstable.slice",
		},
		{
			input:    NewCgroupName(RootCgroupName, "Burstable", "pod-123"),
			expected: "Burstable-pod_123.slice",
		},
		{
			input:    NewCgroupName(RootCgroupName, "test", "a", "b"),
			expected: "test-a-b.slice",
		},
		{
			input:    NewCgroupName(RootCgroupName, "test", "a", "b", "Burstable"),
			expected: "test-a-b-Burstable.slice",
		},
		{
			input:    NewCgroupName(RootCgroupName, "Burstable"),
			expected: "Burstable.slice",
		},
		{
			input:    NewCgroupName(RootCgroupName, "BestEffort", "pod-6c1a4e95-6bb6-11e6-bc26-28d2444e470d"),
			expected: "BestEffort-pod_6c1a4e95_6bb6_11e6_bc26_28d2444e470d.slice",
		},
	}
	for _, testCase := range testCases {
		if actual := path.Base(testCase.input.ToSystemd()); actual != testCase.expected {
			t.Errorf("Unexpected result, input: %v, expected: %v, actual: %v", testCase.input, testCase.expected, actual)
		}
	}
}

func TestCgroupNameToSystemd(t *testing.T) {
	testCases := []struct {
		input    CgroupName
		expected string
	}{
		{
			input:    RootCgroupName,
			expected: "/",
		},
		{
			input:    NewCgroupName(RootCgroupName, "Burstable"),
			expected: "/Burstable.slice",
		},
		{
			input:    NewCgroupName(RootCgroupName, "Burstable", "pod-123"),
			expected: "/Burstable.slice/Burstable-pod_123.slice",
		},
		{
			input:    NewCgroupName(RootCgroupName, "BestEffort", "pod-6c1a4e95-6bb6-11e6-bc26-28d2444e470d"),
			expected: "/BestEffort.slice/BestEffort-pod_6c1a4e95_6bb6_11e6_bc26_28d2444e470d.slice",
		},
		{
			input:    NewCgroupName(RootCgroupName, "kubepods"),
			expected: "/kubepods.slice",
		},
	}
	for _, testCase := range testCases {
		if actual := testCase.input.ToSystemd(); actual != testCase.expected {
			t.Errorf("Unexpected result, input: %v, expected: %v, actual: %v", testCase.input, testCase.expected, actual)
		}
	}
}

func TestCgroupNameToCgroupfs(t *testing.T) {
	testCases := []struct {
		input    CgroupName
		expected string
	}{
		{
			input:    RootCgroupName,
			expected: "/",
		},
		{
			input:    NewCgroupName(RootCgroupName, "Burstable"),
			expected: "/Burstable",
		},
	}
	for _, testCase := range testCases {
		if actual := testCase.input.ToCgroupfs(); actual != testCase.expected {
			t.Errorf("Unexpected result, input: %v, expected: %v, actual: %v", testCase.input, testCase.expected, actual)
		}
	}
}

func TestParseSystemdToCgroupName(t *testing.T) {
	testCases := []struct {
		input    string
		expected CgroupName
	}{
		{
			input:    "/test",
			expected: []string{"test"},
		},
		{
			input:    "/test.slice",
			expected: []string{"test"},
		},
	}

	for _, testCase := range testCases {
		if actual := ParseSystemdToCgroupName(testCase.input); !reflect.DeepEqual(actual, testCase.expected) {
			t.Errorf("Unexpected result, input: %v, expected: %v, actual: %v", testCase.input, testCase.expected, actual)
		}
	}
}

func TestCpuWeightToCPUShares(t *testing.T) {
	testCases := []struct {
		cpuWeight         uint64
		expectedCpuShares uint64
	}{
		{
			cpuWeight:         1,
			expectedCpuShares: 2,
		},
		{
			cpuWeight:         2,
			expectedCpuShares: 28,
		},
		{
			cpuWeight:         3,
			expectedCpuShares: 54,
		},
		{
			cpuWeight:         4,
			expectedCpuShares: 80,
		},
		{
			cpuWeight:         245,
			expectedCpuShares: 6398,
		},
		{
			cpuWeight:         10000,
			expectedCpuShares: 262144,
		},
	}

	for _, testCase := range testCases {
		if actual := cpuWeightToCPUShares(testCase.cpuWeight); actual != testCase.expectedCpuShares {
			t.Errorf("cpuWeight: %v, expectedCpuShares: %v, actualCpuShares: %v",
				testCase.cpuWeight, testCase.expectedCpuShares, actual)
		}
	}
}

func TestCgroupCommonToResources(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	m := &cgroupCommon{
		subsystems: &CgroupSubsystems{
			MountPoints: map[string]string{
				"hugetlb": "/sys/fs/cgroup/hugetlb",
			},
		},
	}
	t.Cleanup(func() {
		m.isUnifiedOverride = nil
	})

	tests := []struct {
		name              string
		resourceConfig    *ResourceConfig
		validateCgroupsV1 func(t *testing.T, res *libcontainercgroups.Resources)
		validateCgroupsV2 func(t *testing.T, res *libcontainercgroups.Resources)
	}{
		{
			name:           "nil config yields default empty resources",
			resourceConfig: nil,
			validateCgroupsV1: func(t *testing.T, res *libcontainercgroups.Resources) {
				require.NotNil(t, res)
				assert.True(t, res.SkipDevices)
				assert.True(t, res.SkipFreezeOnSet)
				assert.Equal(t, int64(0), res.Memory)
				assert.Equal(t, uint64(0), res.CpuShares)
				assert.Equal(t, uint64(0), res.CpuWeight)
			},
			validateCgroupsV2: func(t *testing.T, res *libcontainercgroups.Resources) {
				require.NotNil(t, res)
				assert.True(t, res.SkipDevices)
				assert.True(t, res.SkipFreezeOnSet)
				assert.Equal(t, int64(0), res.Memory)
				assert.Equal(t, uint64(0), res.CpuShares)
				assert.Equal(t, uint64(0), res.CpuWeight)
			},
		},
		{
			name: "memory limits translation",
			resourceConfig: &ResourceConfig{
				Memory: ptr.To[int64](500 * 1024 * 1024),
			},
			validateCgroupsV1: func(t *testing.T, res *libcontainercgroups.Resources) {
				assert.Equal(t, int64(500*1024*1024), res.Memory)
			},
			validateCgroupsV2: func(t *testing.T, res *libcontainercgroups.Resources) {
				assert.Equal(t, int64(500*1024*1024), res.Memory)
			},
		},
		{
			name: "cpu limits period and quota translation",
			resourceConfig: &ResourceConfig{
				CPUPeriod: ptr.To[uint64](100000),
				CPUQuota:  ptr.To[int64](50000),
			},
			validateCgroupsV1: func(t *testing.T, res *libcontainercgroups.Resources) {
				assert.Equal(t, uint64(100000), res.CpuPeriod)
				assert.Equal(t, int64(50000), res.CpuQuota)
			},
			validateCgroupsV2: func(t *testing.T, res *libcontainercgroups.Resources) {
				assert.Equal(t, uint64(100000), res.CpuPeriod)
				assert.Equal(t, int64(50000), res.CpuQuota)
			},
		},
		{
			name: "pids limit translation",
			resourceConfig: &ResourceConfig{
				PidsLimit: ptr.To[int64](1000),
			},
			validateCgroupsV1: func(t *testing.T, res *libcontainercgroups.Resources) {
				assert.Equal(t, ptr.To[int64](1000), res.PidsLimit)
			},
			validateCgroupsV2: func(t *testing.T, res *libcontainercgroups.Resources) {
				assert.Equal(t, ptr.To[int64](1000), res.PidsLimit)
			},
		},
		{
			name: "cpuset translation and string serialization",
			resourceConfig: &ResourceConfig{
				CPUSet: cpuset.New(1, 2, 3),
			},
			validateCgroupsV1: func(t *testing.T, res *libcontainercgroups.Resources) {
				assert.Equal(t, "1-3", res.CpusetCpus)
			},
			validateCgroupsV2: func(t *testing.T, res *libcontainercgroups.Resources) {
				assert.Equal(t, "1-3", res.CpusetCpus)
			},
		},
		{
			name: "cpu shares to weight non-trivial translation",
			resourceConfig: &ResourceConfig{
				CPUShares: ptr.To[uint64](1024),
			},
			validateCgroupsV1: func(t *testing.T, res *libcontainercgroups.Resources) {
				assert.Equal(t, uint64(1024), res.CpuShares)
				assert.Equal(t, uint64(0), res.CpuWeight)
			},
			validateCgroupsV2: func(t *testing.T, res *libcontainercgroups.Resources) {
				assert.Equal(t, uint64(39), res.CpuWeight)
				assert.Equal(t, uint64(0), res.CpuShares)
			},
		},
		{
			name: "unified maps translation conditionally enabled",
			resourceConfig: &ResourceConfig{
				Unified: map[string]string{
					"memory.min": "104857600",
					"memory.low": "209715200",
				},
			},
			validateCgroupsV1: func(t *testing.T, res *libcontainercgroups.Resources) {
				assert.Nil(t, res.Unified)
			},
			validateCgroupsV2: func(t *testing.T, res *libcontainercgroups.Resources) {
				require.NotNil(t, res.Unified)
				assert.Equal(t, "104857600", res.Unified["memory.min"])
				assert.Equal(t, "209715200", res.Unified["memory.low"])
			},
		},
		{
			name: "hugepages limit conversion and host padding",
			resourceConfig: &ResourceConfig{
				HugePageLimit: map[int64]int64{
					2 * 1024 * 1024: 1024 * 1024 * 1024,
				},
			},
			validateCgroupsV1: func(t *testing.T, res *libcontainercgroups.Resources) {
				if len(libcontainercgroups.HugePageSizes()) > 0 {
					require.NotEmpty(t, res.HugetlbLimit)
					found := false
					for _, limit := range res.HugetlbLimit {
						if limit.Pagesize == "2MB" {
							assert.Equal(t, uint64(1024*1024*1024), limit.Limit)
							found = true
						}
					}
					assert.True(t, found, "Should have found translated 2MB hugepage limit under cgroup v1")
				}
			},
			validateCgroupsV2: func(t *testing.T, res *libcontainercgroups.Resources) {
				if len(libcontainercgroups.HugePageSizes()) > 0 {
					require.NotEmpty(t, res.HugetlbLimit)
					found := false
					for _, limit := range res.HugetlbLimit {
						if limit.Pagesize == "2MB" {
							assert.Equal(t, uint64(1024*1024*1024), limit.Limit)
							found = true
						}
					}
					assert.True(t, found, "Should have found translated 2MB hugepage limit under cgroup v2")
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// 1. Validate cgroup v1 mode
			m.isUnifiedOverride = ptr.To(false)
			resV1 := m.toResources(logger, tc.resourceConfig)
			tc.validateCgroupsV1(t, resV1)

			// 2. Validate cgroup v2 mode	
			m.isUnifiedOverride = ptr.To(true)
			resV2 := m.toResources(logger, tc.resourceConfig)
			tc.validateCgroupsV2(t, resV2)
		})
	}
}
