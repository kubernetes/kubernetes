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

func TestCgroupNsdelegateEnabled(t *testing.T) {
	tests := []struct {
		name      string
		mountinfo string
		want      bool
		wantErr   bool
	}{
		{
			name:      "nsdelegate present",
			mountinfo: "35 25 0:30 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:9 - cgroup2 cgroup2 rw,nsdelegate,memory_recursiveprot\n",
			want:      true,
		},
		{
			name:      "nsdelegate absent",
			mountinfo: "35 25 0:30 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:9 - cgroup2 cgroup2 rw,memory_recursiveprot\n",
			want:      false,
		},
		{
			name:      "nsdelegate only in the per-mount options",
			mountinfo: "35 25 0:30 / /sys/fs/cgroup rw,nsdelegate shared:9 - cgroup2 cgroup2 rw\n",
			want:      false,
		},
		{
			name: "nsdelegate on a cgroup v1 mount only",
			mountinfo: "35 25 0:30 / /sys/fs/cgroup ro,nosuid,nodev,noexec shared:9 - tmpfs tmpfs ro,mode=755\n" +
				"36 35 0:31 / /sys/fs/cgroup/memory rw,relatime shared:10 - cgroup cgroup rw,nsdelegate,memory\n",
			want: false,
		},
		{
			name: "nsdelegate on another cgroup2 mount only",
			mountinfo: "34 25 0:29 / /run rw,nosuid,nodev shared:8 - tmpfs tmpfs rw,size=1024k\n" +
				"35 25 0:30 / /sys/fs/cgroup rw,relatime shared:9 - cgroup2 cgroup2 rw\n" +
				"36 25 0:30 / /host/cgroup rw,relatime shared:9 - cgroup2 cgroup2 rw,nsdelegate\n",
			want: false,
		},
		{
			name: "later mount at the same path has nsdelegate",
			mountinfo: "35 25 0:30 / /sys/fs/cgroup rw,relatime shared:9 - cgroup2 cgroup2 rw\n" +
				"36 35 0:31 / /sys/fs/cgroup rw,relatime shared:10 - cgroup2 cgroup2 rw,nsdelegate\n",
			want: true,
		},
		{
			name: "later mount at the same path lacks nsdelegate",
			mountinfo: "35 25 0:30 / /sys/fs/cgroup rw,relatime shared:9 - cgroup2 cgroup2 rw,nsdelegate\n" +
				"36 35 0:31 / /sys/fs/cgroup rw,relatime shared:10 - cgroup2 cgroup2 rw\n",
			want: false,
		},
		{
			name:      "no mount at /sys/fs/cgroup",
			mountinfo: "34 25 0:29 / /run rw,nosuid,nodev shared:8 - tmpfs tmpfs rw,size=1024k\n",
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mountInfoPath := filepath.Join(t.TempDir(), "mountinfo")
			if err := os.WriteFile(mountInfoPath, []byte(tt.mountinfo), 0o600); err != nil {
				t.Fatal(err)
			}
			got, err := cgroupNsdelegateEnabled(mountInfoPath)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("cgroupNsdelegateEnabled should fail for mountinfo %q", tt.mountinfo)
				}
				return
			}
			if err != nil {
				t.Fatalf("cgroupNsdelegateEnabled should read the fixture, got error: %v", err)
			}
			if got != tt.want {
				t.Errorf("nsdelegate should be reported as %v for mountinfo %q, got %v", tt.want, tt.mountinfo, got)
			}
		})
	}

	t.Run("unreadable mountinfo", func(t *testing.T) {
		if _, err := cgroupNsdelegateEnabled(filepath.Join(t.TempDir(), "missing")); err == nil {
			t.Error("cgroupNsdelegateEnabled should fail when mountinfo cannot be read")
		}
	})
}
