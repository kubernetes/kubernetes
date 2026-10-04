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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/cpuset"
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
		m.isCgroup2UnifiedModeOverride = nil
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
				Memory: new(int64(500 * 1024 * 1024)),
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
				CPUPeriod: new(uint64(100000)),
				CPUQuota:  new(int64(50000)),
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
				PidsLimit: new(int64(1000)),
			},
			validateCgroupsV1: func(t *testing.T, res *libcontainercgroups.Resources) {
				assert.Equal(t, new(int64(1000)), res.PidsLimit)
			},
			validateCgroupsV2: func(t *testing.T, res *libcontainercgroups.Resources) {
				assert.Equal(t, new(int64(1000)), res.PidsLimit)
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
				CPUShares: new(uint64(1024)),
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
			m.isCgroup2UnifiedModeOverride = new(bool)
			resV1 := m.toResources(logger, tc.resourceConfig)
			tc.validateCgroupsV1(t, resV1)

			// 2. Validate cgroup v2 mode
			m.isCgroup2UnifiedModeOverride = new(true)
			resV2 := m.toResources(logger, tc.resourceConfig)
			tc.validateCgroupsV2(t, resV2)
		})
	}
}

func TestCgroupManagerGetCgroupConfig(t *testing.T) {
	libcontainercgroups.TestMode = true
	t.Cleanup(func() {
		libcontainercgroups.TestMode = false
	})

	cgroupName := CgroupName([]string{"test-pod"})
	adaptedName := cgroupName.ToCgroupfs()

	tests := []struct {
		name     string
		resource v1.ResourceName

		setupFilesV1 func(t *testing.T, podDir string)
		setupFilesV2 func(t *testing.T, podDir string)

		expectErr   bool
		mountPoints map[string]string
		expectedCfg *ResourceConfig
	}{
		{
			name:     "valid CPU quota, period, and shares",
			resource: v1.ResourceCPU,
			setupFilesV1: func(t *testing.T, podDir string) {
				require.NoError(t, os.WriteFile(filepath.Join(podDir, "cpu.cfs_quota_us"), []byte("50000\n"), 0644))
				require.NoError(t, os.WriteFile(filepath.Join(podDir, "cpu.cfs_period_us"), []byte("100000\n"), 0644))
				require.NoError(t, os.WriteFile(filepath.Join(podDir, "cpu.shares"), []byte("998\n"), 0644))
			},
			setupFilesV2: func(t *testing.T, podDir string) {
				require.NoError(t, os.WriteFile(filepath.Join(podDir, cgroupv2CpuMaxFile), []byte("50000 100000\n"), 0644))
				require.NoError(t, os.WriteFile(filepath.Join(podDir, cgroupv2CpuWeightFile), []byte("39\n"), 0644))
			},
			expectedCfg: &ResourceConfig{
				CPUQuota:  new(int64(50000)),
				CPUPeriod: new(uint64(100000)),
				CPUShares: new(uint64(998)),
			},
		},
		{
			name:     "unlimited CPU quota ('max')",
			resource: v1.ResourceCPU,
			setupFilesV1: func(t *testing.T, podDir string) {
				require.NoError(t, os.WriteFile(filepath.Join(podDir, "cpu.cfs_quota_us"), []byte("-1\n"), 0644))
				require.NoError(t, os.WriteFile(filepath.Join(podDir, "cpu.cfs_period_us"), []byte("100000\n"), 0644))
				require.NoError(t, os.WriteFile(filepath.Join(podDir, "cpu.shares"), []byte("2597\n"), 0644))
			},
			setupFilesV2: func(t *testing.T, podDir string) {
				require.NoError(t, os.WriteFile(filepath.Join(podDir, cgroupv2CpuMaxFile), []byte("max 100000\n"), 0644))
				require.NoError(t, os.WriteFile(filepath.Join(podDir, cgroupv2CpuWeightFile), []byte("100\n"), 0644))
			},
			expectedCfg: &ResourceConfig{
				CPUQuota:  new(int64(-1)),
				CPUPeriod: new(uint64(100000)),
				CPUShares: new(uint64(2597)),
			},
		},
		{
			name:     "valid memory limit",
			resource: v1.ResourceMemory,
			setupFilesV1: func(t *testing.T, podDir string) {
				require.NoError(t, os.WriteFile(filepath.Join(podDir, cgroupv1MemLimitFile), []byte("209715200\n"), 0644))
			},
			setupFilesV2: func(t *testing.T, podDir string) {
				require.NoError(t, os.WriteFile(filepath.Join(podDir, cgroupv2MemLimitFile), []byte("209715200\n"), 0644))
			},
			expectedCfg: &ResourceConfig{
				Memory: new(int64(209715200)),
			},
		},
		{
			name:     "memory limit max returns -1 limit",
			resource: v1.ResourceMemory,
			setupFilesV1: func(t *testing.T, podDir string) {
				require.NoError(t, os.WriteFile(filepath.Join(podDir, cgroupv1MemLimitFile), []byte("max\n"), 0644))
			},
			setupFilesV2: func(t *testing.T, podDir string) {
				require.NoError(t, os.WriteFile(filepath.Join(podDir, cgroupv2MemLimitFile), []byte("max\n"), 0644))
			},
			expectedCfg: &ResourceConfig{
				Memory: new(int64(-1)),
			},
		},
		{
			name:        "missing cgroup mount point returns error",
			resource:    v1.ResourceCPU,
			expectErr:   true,
			mountPoints: map[string]string{"memory": ""},
		},
		{
			name:        "unsupported resource returns error",
			resource:    v1.ResourceStorage,
			expectErr:   true,
			mountPoints: map[string]string{"storage": ""},
		},
		{
			name:     "malformed CPU quota integer returns error",
			resource: v1.ResourceCPU,
			setupFilesV1: func(t *testing.T, podDir string) {
				require.NoError(t, os.WriteFile(filepath.Join(podDir, "cpu.cfs_quota_us"), []byte("invalid\n"), 0644))
			},
			setupFilesV2: func(t *testing.T, podDir string) {
				require.NoError(t, os.WriteFile(filepath.Join(podDir, cgroupv2CpuMaxFile), []byte("invalid 100000\n"), 0644))
			},
			expectErr: true,
		},
		{
			name:     "malformed cpu.max single token returns error under v2",
			resource: v1.ResourceCPU,
			setupFilesV2: func(t *testing.T, podDir string) {
				require.NoError(t, os.WriteFile(filepath.Join(podDir, cgroupv2CpuMaxFile), []byte("50000\n"), 0644))
			},
			expectErr: true,
		},
		{
			name:     "malformed cpu.max invalid period integer returns error under v2",
			resource: v1.ResourceCPU,
			setupFilesV2: func(t *testing.T, podDir string) {
				require.NoError(t, os.WriteFile(filepath.Join(podDir, cgroupv2CpuMaxFile), []byte("50000 invalid\n"), 0644))
			},
			expectErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			runners := []struct {
				version    string
				setupFiles func(t *testing.T, podDir string)
				manager    func(subsystems *CgroupSubsystems) CgroupManager
			}{
				{
					version:    "v1",
					setupFiles: tc.setupFilesV1,
					manager: func(subsystems *CgroupSubsystems) CgroupManager {
						return &cgroupV1impl{cgroupCommon: cgroupCommon{subsystems: subsystems}}
					},
				},
				{
					version:    "v2",
					setupFiles: tc.setupFilesV2,
					manager: func(subsystems *CgroupSubsystems) CgroupManager {
						return &cgroupV2impl{cgroupCommon: cgroupCommon{subsystems: subsystems}}
					},
				},
			}

			for _, runner := range runners {
				t.Run(runner.version, func(t *testing.T) {
					tempDir := t.TempDir()
					podDir := filepath.Join(tempDir, adaptedName)
					require.NoError(t, os.MkdirAll(podDir, 0755))
					if runner.setupFiles != nil {
						runner.setupFiles(t, podDir)
					}
					mounts := map[string]string{"cpu": tempDir, "memory": tempDir}
					if tc.mountPoints != nil {
						mounts = make(map[string]string, len(tc.mountPoints))
						for k := range tc.mountPoints {
							mounts[k] = tempDir
						}
					}
					m := runner.manager(&CgroupSubsystems{MountPoints: mounts})
					cfg, err := m.GetCgroupConfig(cgroupName, tc.resource)
					if tc.expectErr {
						require.Errorf(t, err, "expected error for %s in %s", tc.name, runner.version)
					} else {
						require.NoErrorf(t, err, "unexpected error for %s in %s", tc.name, runner.version)
						assert.Equalf(t, tc.expectedCfg, cfg, "mismatch for %s in %s", tc.name, runner.version)
					}
				})
			}
		})
	}
}
