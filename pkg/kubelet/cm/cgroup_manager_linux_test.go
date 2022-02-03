//go:build linux
// +build linux

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
	"fmt"
	"path"
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"
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

func TestIsSystemdStyleName(t *testing.T) {
	tc := []struct {
		input    string
		expected bool
	}{
		{
			input:    "/test",
			expected: false,
		},
		{
			input:    "/test.slice",
			expected: true,
		},
	}

	for _, c := range tc {
		t.Run(fmt.Sprintf("%q", c.input), func(t *testing.T) {
			result := IsSystemdStyleName(c.input)
			require.Equal(t, c.expected, result)
		})
	}
}

func newTestCgroupManger(t *testing.T, cgroupDriver string) CgroupManager {
	t.Helper()
	subsystems, err := GetCgroupSubsystems()
	require.NoError(t, err)

	mgr := NewCgroupManager(subsystems, cgroupDriver)
	return mgr
}

func TestCgroupManager_Name(t *testing.T) {
	tc := []struct {
		name         string
		cgroupDriver string
		input        CgroupName
		expected     string
	}{
		{
			name:         "with_cgroupfs_adapter_returns_cgroupfs_paths",
			cgroupDriver: string(libcontainerCgroupfs),
			input:        NewCgroupName(RootCgroupName, "Burstable"),
			expected:     "/Burstable",
		},
		{
			name:         "with_systemd_adapter_returns_systemd_paths",
			cgroupDriver: string(libcontainerSystemd),
			input:        NewCgroupName(RootCgroupName, "Burstable"),
			expected:     "/Burstable.slice",
		},
	}

	for _, c := range tc {
		t.Run(c.name, func(t *testing.T) {
			mgr := newTestCgroupManger(t, c.cgroupDriver)
			require.Equal(t, c.expected, mgr.Name(c.input))
		})
	}
}

func TestCgroupManager_ParseName(t *testing.T) {
	tc := []struct {
		name         string
		cgroupDriver string
		input        string
		expected     CgroupName
	}{
		{
			name:         "with_cgroupfs_adapter_parses_cgroupfs_paths",
			cgroupDriver: string(libcontainerCgroupfs),
			input:        "/Burstable",
			expected:     NewCgroupName(RootCgroupName, "Burstable"),
		},
		{
			name:         "with_systemd_adapter_parses_systemd_paths",
			cgroupDriver: string(libcontainerSystemd),
			input:        "/Burstable.slice",
			expected:     NewCgroupName(RootCgroupName, "Burstable"),
		},
	}

	for _, c := range tc {
		t.Run(c.name, func(t *testing.T) {
			mgr := newTestCgroupManger(t, c.cgroupDriver)
			require.Equal(t, c.expected, mgr.CgroupName(c.input))
		})
	}
}
