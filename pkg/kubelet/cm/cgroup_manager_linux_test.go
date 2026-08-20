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
	"os"
	"path"
	"path/filepath"
	"reflect"
	"testing"

	libcontainercgroups "github.com/opencontainers/cgroups"
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

func TestReadCgroupMemoryConfig(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "cgroup_memory_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	limitFile := "memory.max"
	limitVal := "2097152"
	if err := os.WriteFile(filepath.Join(tempDir, limitFile), []byte(limitVal), 0644); err != nil {
		t.Fatalf("failed to write limit file: %v", err)
	}

	minVal := "102400"
	lowVal := "204800"
	highVal := "409600"
	if err := os.WriteFile(filepath.Join(tempDir, "memory.min"), []byte(minVal), 0644); err != nil {
		t.Fatalf("failed to write memory.min: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tempDir, "memory.low"), []byte(lowVal), 0644); err != nil {
		t.Fatalf("failed to write memory.low: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tempDir, "memory.high"), []byte(highVal), 0644); err != nil {
		t.Fatalf("failed to write memory.high: %v", err)
	}

	rc, err := readCgroupMemoryConfig(tempDir, limitFile)
	if err != nil {
		t.Fatalf("failed to read cgroup memory config: %v", err)
	}

	if rc.Memory == nil || *rc.Memory != 2097152 {
		t.Errorf("expected Memory limit to be 2097152, got %v", rc.Memory)
	}

	if libcontainercgroups.IsCgroup2UnifiedMode() {
		if rc.Unified == nil {
			t.Errorf("expected Unified map to be populated in cgroup v2 mode")
		} else {
			if val, ok := rc.Unified["memory.min"]; !ok || val != minVal {
				t.Errorf("expected memory.min to be %s, got %s", minVal, val)
			}
			if val, ok := rc.Unified["memory.low"]; !ok || val != lowVal {
				t.Errorf("expected memory.low to be %s, got %s", lowVal, val)
			}
			if val, ok := rc.Unified["memory.high"]; !ok || val != highVal {
				t.Errorf("expected memory.high to be %s, got %s", highVal, val)
			}
		}
	} else {
		if rc.Unified != nil {
			t.Errorf("expected Unified map to be nil in cgroup v1 mode")
		}
	}
}
