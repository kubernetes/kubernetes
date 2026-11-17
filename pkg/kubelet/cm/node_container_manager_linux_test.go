//go:build linux

/*
Copyright 2017 The Kubernetes Authors.

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
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	ktesting "k8s.io/klog/v2/ktesting"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	"k8s.io/utils/ptr"
)

func TestUpdateNodeAllocatableCgroup(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	readFailure := errors.New("read failed")

	tests := []struct {
		name string
		// version is the cgroup manager version under test.
		version int
		// memory is the requested Node Allocatable memory limit.
		memory *int64
		// cpuShares is an unrelated setting that must not be suppressed.
		cpuShares *uint64
		// usage is what MemoryUsage reports for the cgroup.
		usage    int64
		usageErr error
		// wantErr is whether the helper reports the limit as not yet applied.
		wantErr bool
		// wantUpdates is how many Update calls the helper is expected to make.
		wantUpdates int
		// wantAppliedMemory is the memory limit recorded on the last Update.
		wantAppliedMemory *int64
		// wantAppliedMemoryNil asserts the last Update carried no memory limit.
		wantAppliedMemoryNil bool
		// wantAppliedCPUShares asserts the last Update carried this CPU shares value.
		wantAppliedCPUShares *uint64
	}{
		{
			name:              "v2 usage at limit",
			version:           2,
			memory:            ptr.To[int64](100),
			usage:             100,
			wantUpdates:       1,
			wantAppliedMemory: ptr.To[int64](100),
		},
		{
			// The requested limit is unsafe to write, so the helper clamps to the
			// current usage: the write is safe and it creates the pressure the
			// retry loop depends on. The error keeps the retry alive.
			name:              "v2 usage above limit clamps to usage",
			version:           2,
			memory:            ptr.To[int64](100),
			usage:             101,
			wantErr:           true,
			wantUpdates:       1,
			wantAppliedMemory: ptr.To[int64](101),
		},
		{
			// Zero means "unset" to both cgroup v2 backends, so no memory.max
			// write happens and positive usage must not block the update.
			name:              "v2 zero memory is unset, not a limit",
			version:           2,
			memory:            ptr.To[int64](0),
			usage:             101,
			wantUpdates:       1,
			wantAppliedMemory: ptr.To[int64](0),
		},
		{
			name:                 "v2 nil memory skips the usage guard",
			version:              2,
			memory:               nil,
			cpuShares:            ptr.To[uint64](1024),
			usage:                101,
			wantUpdates:          1,
			wantAppliedMemoryNil: true,
			wantAppliedCPUShares: ptr.To[uint64](1024),
		},
		{
			name:              "v1 keeps existing update behavior",
			version:           1,
			memory:            ptr.To[int64](100),
			usage:             101,
			wantUpdates:       1,
			wantAppliedMemory: ptr.To[int64](100),
		},
		{
			// A failed memory read must not hold back independent settings.
			name:                 "usage read failure still applies other settings",
			version:              2,
			memory:               ptr.To[int64](100),
			cpuShares:            ptr.To[uint64](1024),
			usageErr:             readFailure,
			wantErr:              true,
			wantUpdates:          1,
			wantAppliedMemoryNil: true,
			wantAppliedCPUShares: ptr.To[uint64](1024),
		},
		{
			// Deferring memory must not drop the CPU share correction.
			name:                 "usage above limit still applies cpu shares",
			version:              2,
			memory:               ptr.To[int64](100),
			cpuShares:            ptr.To[uint64](1024),
			usage:                101,
			wantErr:              true,
			wantUpdates:          1,
			wantAppliedMemory:    ptr.To[int64](101),
			wantAppliedCPUShares: ptr.To[uint64](1024),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			manager := &fakeCgroupManager{version: tc.version, usage: tc.usage, usageErr: tc.usageErr}
			cm := &containerManagerImpl{cgroupManager: manager}
			config := &CgroupConfig{
				Name: CgroupName{"kubepods"},
				ResourceParameters: &ResourceConfig{
					Memory:    tc.memory,
					CPUShares: tc.cpuShares,
				},
			}

			err := cm.updateNodeAllocatableCgroup(logger, config)
			if tc.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}

			require.Len(t, manager.updates, tc.wantUpdates)
			// The caller's configuration must stay untouched across retries.
			require.Same(t, tc.memory, config.ResourceParameters.Memory)

			if tc.wantUpdates == 0 {
				return
			}
			applied := manager.updates[len(manager.updates)-1].ResourceParameters
			require.NotNil(t, applied)
			if tc.wantAppliedMemoryNil {
				assert.Nil(t, applied.Memory)
			} else {
				require.NotNil(t, applied.Memory)
				assert.Equal(t, *tc.wantAppliedMemory, *applied.Memory)
			}
			if tc.wantAppliedCPUShares != nil {
				require.NotNil(t, applied.CPUShares)
				assert.Equal(t, *tc.wantAppliedCPUShares, *applied.CPUShares)
			}
		})
	}
}

func TestNodeAllocatableReservationForScheduling(t *testing.T) {
	memoryEvictionThreshold := resource.MustParse("100Mi")
	cpuMemCases := []struct {
		kubeReserved   v1.ResourceList
		systemReserved v1.ResourceList
		expected       v1.ResourceList
		capacity       v1.ResourceList
		hardThreshold  evictionapi.ThresholdValue
	}{
		{
			kubeReserved:   getResourceList("100m", "100Mi"),
			systemReserved: getResourceList("50m", "50Mi"),
			capacity:       getResourceList("10", "10Gi"),
			expected:       getResourceList("150m", "150Mi"),
		},
		{
			kubeReserved:   getResourceList("100m", "100Mi"),
			systemReserved: getResourceList("50m", "50Mi"),
			hardThreshold: evictionapi.ThresholdValue{
				Quantity: &memoryEvictionThreshold,
			},
			capacity: getResourceList("10", "10Gi"),
			expected: getResourceList("150m", "250Mi"),
		},
		{
			kubeReserved:   getResourceList("100m", "100Mi"),
			systemReserved: getResourceList("50m", "50Mi"),
			capacity:       getResourceList("10", "10Gi"),
			hardThreshold: evictionapi.ThresholdValue{
				Percentage: 0.05,
			},
			expected: getResourceList("150m", "694157320"),
		},

		{
			kubeReserved:   v1.ResourceList{},
			systemReserved: v1.ResourceList{},
			capacity:       getResourceList("10", "10Gi"),
			expected:       getResourceList("", ""),
		},
		{
			kubeReserved:   getResourceList("", "100Mi"),
			systemReserved: getResourceList("50m", "50Mi"),
			capacity:       getResourceList("10", "10Gi"),
			expected:       getResourceList("50m", "150Mi"),
		},

		{
			kubeReserved:   getResourceList("50m", "100Mi"),
			systemReserved: getResourceList("", "50Mi"),
			capacity:       getResourceList("10", "10Gi"),
			expected:       getResourceList("50m", "150Mi"),
		},
		{
			kubeReserved:   getResourceList("", "100Mi"),
			systemReserved: getResourceList("", "50Mi"),
			capacity:       getResourceList("10", ""),
			expected:       getResourceList("", "150Mi"),
		},
	}
	for idx, tc := range cpuMemCases {
		nc := NodeConfig{
			NodeAllocatableConfig: NodeAllocatableConfig{
				KubeReserved:   tc.kubeReserved,
				SystemReserved: tc.systemReserved,
				HardEvictionThresholds: []evictionapi.Threshold{
					{
						Signal:   evictionapi.SignalMemoryAvailable,
						Operator: evictionapi.OpLessThan,
						Value:    tc.hardThreshold,
					},
				},
			},
		}
		cm := &containerManagerImpl{
			NodeConfig: nc,
			capacity:   tc.capacity,
		}
		for k, v := range cm.GetNodeAllocatableReservation() {
			expected, exists := tc.expected[k]
			assert.True(t, exists, "test case %d expected resource %q", idx+1, k)
			assert.Equal(t, expected.MilliValue(), v.MilliValue(), "test case %d failed for resource %q", idx+1, k)
		}
	}

	ephemeralStorageEvictionThreshold := resource.MustParse("100Mi")
	ephemeralStorageTestCases := []struct {
		kubeReserved  v1.ResourceList
		expected      v1.ResourceList
		capacity      v1.ResourceList
		hardThreshold evictionapi.ThresholdValue
	}{
		{
			kubeReserved: getEphemeralStorageResourceList("100Mi"),
			capacity:     getEphemeralStorageResourceList("10Gi"),
			expected:     getEphemeralStorageResourceList("100Mi"),
		},
		{
			kubeReserved: getEphemeralStorageResourceList("100Mi"),
			hardThreshold: evictionapi.ThresholdValue{
				Quantity: &ephemeralStorageEvictionThreshold,
			},
			capacity: getEphemeralStorageResourceList("10Gi"),
			expected: getEphemeralStorageResourceList("200Mi"),
		},
		{
			kubeReserved: getEphemeralStorageResourceList("150Mi"),
			capacity:     getEphemeralStorageResourceList("10Gi"),
			hardThreshold: evictionapi.ThresholdValue{
				Percentage: 0.05,
			},
			expected: getEphemeralStorageResourceList("694157320"),
		},

		{
			kubeReserved: v1.ResourceList{},
			capacity:     getEphemeralStorageResourceList("10Gi"),
			expected:     getEphemeralStorageResourceList(""),
		},
	}
	for idx, tc := range ephemeralStorageTestCases {
		nc := NodeConfig{
			NodeAllocatableConfig: NodeAllocatableConfig{
				KubeReserved: tc.kubeReserved,
				HardEvictionThresholds: []evictionapi.Threshold{
					{
						Signal:   evictionapi.SignalNodeFsAvailable,
						Operator: evictionapi.OpLessThan,
						Value:    tc.hardThreshold,
					},
				},
			},
		}
		cm := &containerManagerImpl{
			NodeConfig: nc,
			capacity:   tc.capacity,
		}
		for k, v := range cm.GetNodeAllocatableReservation() {
			expected, exists := tc.expected[k]
			assert.True(t, exists, "test case %d expected resource %q", idx+1, k)
			assert.Equal(t, expected.MilliValue(), v.MilliValue(), "test case %d failed for resource %q", idx+1, k)
		}
	}
}

func TestNodeAllocatableForEnforcement(t *testing.T) {
	memoryEvictionThreshold := resource.MustParse("100Mi")
	testCases := []struct {
		kubeReserved   v1.ResourceList
		systemReserved v1.ResourceList
		capacity       v1.ResourceList
		expected       v1.ResourceList
		hardThreshold  evictionapi.ThresholdValue
	}{
		{
			kubeReserved:   getResourceList("100m", "100Mi"),
			systemReserved: getResourceList("50m", "50Mi"),
			capacity:       getResourceList("10", "10Gi"),
			expected:       getResourceList("9850m", "10090Mi"),
		},
		{
			kubeReserved:   getResourceList("100m", "100Mi"),
			systemReserved: getResourceList("50m", "50Mi"),
			hardThreshold: evictionapi.ThresholdValue{
				Quantity: &memoryEvictionThreshold,
			},
			capacity: getResourceList("10", "10Gi"),
			expected: getResourceList("9850m", "10090Mi"),
		},
		{
			kubeReserved:   getResourceList("100m", "100Mi"),
			systemReserved: getResourceList("50m", "50Mi"),
			hardThreshold: evictionapi.ThresholdValue{
				Percentage: 0.05,
			},
			capacity: getResourceList("10", "10Gi"),
			expected: getResourceList("9850m", "10090Mi"),
		},

		{
			kubeReserved:   v1.ResourceList{},
			systemReserved: v1.ResourceList{},
			capacity:       getResourceList("10", "10Gi"),
			expected:       getResourceList("10", "10Gi"),
		},
		{
			kubeReserved:   getResourceList("", "100Mi"),
			systemReserved: getResourceList("50m", "50Mi"),
			capacity:       getResourceList("10", "10Gi"),
			expected:       getResourceList("9950m", "10090Mi"),
		},

		{
			kubeReserved:   getResourceList("50m", "100Mi"),
			systemReserved: getResourceList("", "50Mi"),
			capacity:       getResourceList("10", "10Gi"),
			expected:       getResourceList("9950m", "10090Mi"),
		},
		{
			kubeReserved:   getResourceList("", "100Mi"),
			systemReserved: getResourceList("", "50Mi"),
			capacity:       getResourceList("10", ""),
			expected:       getResourceList("10", ""),
		},
	}
	for idx, tc := range testCases {
		nc := NodeConfig{
			NodeAllocatableConfig: NodeAllocatableConfig{
				KubeReserved:   tc.kubeReserved,
				SystemReserved: tc.systemReserved,
				HardEvictionThresholds: []evictionapi.Threshold{
					{
						Signal:   evictionapi.SignalMemoryAvailable,
						Operator: evictionapi.OpLessThan,
						Value:    tc.hardThreshold,
					},
				},
			},
		}
		cm := &containerManagerImpl{
			NodeConfig: nc,
			capacity:   tc.capacity,
		}
		for k, v := range cm.GetNodeAllocatableAbsolute() {
			expected, exists := tc.expected[k]
			assert.True(t, exists)
			assert.Equal(t, expected.MilliValue(), v.MilliValue(), "test case %d failed for resource %q", idx+1, k)
		}
	}
}

func TestNodeAllocatableInputValidation(t *testing.T) {
	memoryEvictionThreshold := resource.MustParse("100Mi")
	highMemoryEvictionThreshold := resource.MustParse("2Gi")
	cpuMemTestCases := []struct {
		kubeReserved         v1.ResourceList
		systemReserved       v1.ResourceList
		capacity             v1.ResourceList
		hardThreshold        evictionapi.ThresholdValue
		invalidConfiguration bool
	}{
		{
			kubeReserved:   getResourceList("100m", "100Mi"),
			systemReserved: getResourceList("50m", "50Mi"),
			capacity:       getResourceList("10", "10Gi"),
		},
		{
			kubeReserved:   getResourceList("100m", "100Mi"),
			systemReserved: getResourceList("50m", "50Mi"),
			hardThreshold: evictionapi.ThresholdValue{
				Quantity: &memoryEvictionThreshold,
			},
			capacity: getResourceList("10", "10Gi"),
		},
		{
			kubeReserved:   getResourceList("100m", "100Mi"),
			systemReserved: getResourceList("50m", "50Mi"),
			hardThreshold: evictionapi.ThresholdValue{
				Percentage: 0.05,
			},
			capacity: getResourceList("10", "10Gi"),
		},
		{
			kubeReserved:   v1.ResourceList{},
			systemReserved: v1.ResourceList{},
			capacity:       getResourceList("10", "10Gi"),
		},
		{
			kubeReserved:   getResourceList("", "100Mi"),
			systemReserved: getResourceList("50m", "50Mi"),
			capacity:       getResourceList("10", "10Gi"),
		},
		{
			kubeReserved:   getResourceList("50m", "100Mi"),
			systemReserved: getResourceList("", "50Mi"),
			capacity:       getResourceList("10", "10Gi"),
		},
		{
			kubeReserved:   getResourceList("", "100Mi"),
			systemReserved: getResourceList("", "50Mi"),
			capacity:       getResourceList("10", ""),
		},
		{
			kubeReserved:   getResourceList("5", "10Gi"),
			systemReserved: getResourceList("5", "10Gi"),
			hardThreshold: evictionapi.ThresholdValue{
				Quantity: &highMemoryEvictionThreshold,
			},
			capacity:             getResourceList("10", "11Gi"),
			invalidConfiguration: true,
		},
	}
	for _, tc := range cpuMemTestCases {
		nc := NodeConfig{
			NodeAllocatableConfig: NodeAllocatableConfig{
				KubeReserved:   tc.kubeReserved,
				SystemReserved: tc.systemReserved,
				HardEvictionThresholds: []evictionapi.Threshold{
					{
						Signal:   evictionapi.SignalMemoryAvailable,
						Operator: evictionapi.OpLessThan,
						Value:    tc.hardThreshold,
					},
				},
			},
		}
		cm := &containerManagerImpl{
			NodeConfig: nc,
			capacity:   tc.capacity,
		}
		err := cm.validateNodeAllocatable()
		if err == nil && tc.invalidConfiguration {
			t.Fatalf("Expected invalid node allocatable configuration")
		} else if err != nil && !tc.invalidConfiguration {
			t.Fatalf("Expected valid node allocatable configuration: %v", err)
		}
	}

	ephemeralStorageEvictionThreshold := resource.MustParse("100Mi")
	ephemeralStorageTestCases := []struct {
		kubeReserved         v1.ResourceList
		capacity             v1.ResourceList
		hardThreshold        evictionapi.ThresholdValue
		invalidConfiguration bool
	}{
		{
			kubeReserved: getEphemeralStorageResourceList("100Mi"),
			capacity:     getEphemeralStorageResourceList("500Mi"),
		},
		{
			kubeReserved: getEphemeralStorageResourceList("20Gi"),
			hardThreshold: evictionapi.ThresholdValue{
				Quantity: &ephemeralStorageEvictionThreshold,
			},
			capacity:             getEphemeralStorageResourceList("20Gi"),
			invalidConfiguration: true,
		},
	}
	for _, tc := range ephemeralStorageTestCases {
		nc := NodeConfig{
			NodeAllocatableConfig: NodeAllocatableConfig{
				KubeReserved: tc.kubeReserved,
				HardEvictionThresholds: []evictionapi.Threshold{
					{
						Signal:   evictionapi.SignalNodeFsAvailable,
						Operator: evictionapi.OpLessThan,
						Value:    tc.hardThreshold,
					},
				},
			},
		}
		cm := &containerManagerImpl{
			NodeConfig: nc,
			capacity:   tc.capacity,
		}
		err := cm.validateNodeAllocatable()
		if err == nil && tc.invalidConfiguration {
			t.Fatalf("Expected invalid node allocatable configuration")
		} else if err != nil && !tc.invalidConfiguration {
			t.Fatalf("Expected valid node allocatable configuration: %v", err)
		}
	}
}

// getEphemeralStorageResourceList returns a ResourceList with the
// specified ephemeral storage resource values
func getEphemeralStorageResourceList(storage string) v1.ResourceList {
	res := v1.ResourceList{}
	if storage != "" {
		res[v1.ResourceEphemeralStorage] = resource.MustParse(storage)
	}
	return res
}

func TestNodeRefFromNode(t *testing.T) {
	testCases := []struct {
		name     string
		nodeName string
		expected *v1.ObjectReference
	}{
		{
			name:     "normal node name",
			nodeName: "test-node",
			expected: &v1.ObjectReference{
				APIVersion: "v1",
				Kind:       "Node",
				Name:       "test-node",
				Namespace:  "",
			},
		},
		{
			name:     "empty node name",
			nodeName: "",
			expected: &v1.ObjectReference{
				APIVersion: "v1",
				Kind:       "Node",
				Name:       "",
				UID:        "",
				Namespace:  "",
			},
		},
		{
			name:     "node name with special characters",
			nodeName: "test-node-123.domain.local",
			expected: &v1.ObjectReference{
				APIVersion: "v1",
				Kind:       "Node",
				Name:       "test-node-123.domain.local",
				Namespace:  "",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := nodeRefFromNode(tc.nodeName)

			assert.Equal(t, tc.expected, result, "test case %q failed", tc.name)
		})
	}
}

func TestGetCgroupConfig(t *testing.T) {
	cases := []struct {
		name                  string
		resourceList          v1.ResourceList
		compressibleResources bool
		checks                func(*ResourceConfig, *testing.T)
	}{
		{
			name:                  "Nil resource list",
			resourceList:          nil,
			compressibleResources: false,
			checks: func(actual *ResourceConfig, t *testing.T) {
				assert.Nil(t, actual)
			},
		},
		{
			name: "Compressible resources only",
			resourceList: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("200Mi"),
			},
			compressibleResources: true,
			checks: func(actual *ResourceConfig, t *testing.T) {
				assert.NotNil(t, actual.CPUShares)
				assert.Nil(t, actual.Memory)
				assert.Nil(t, actual.PidsLimit)
				assert.Nil(t, actual.HugePageLimit)
			},
		},
		{
			name: "Memory only",
			resourceList: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("200Mi"),
			},
			compressibleResources: false,
			checks: func(actual *ResourceConfig, t *testing.T) {
				assert.NotNil(t, actual.Memory)
				assert.Nil(t, actual.CPUShares)
			},
		},
		{
			name: "Memory and CPU without compressible resources",
			resourceList: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("200Mi"),
			},
			compressibleResources: false,
			checks: func(actual *ResourceConfig, t *testing.T) {
				assert.NotNil(t, actual.Memory)
				assert.NotNil(t, actual.CPUShares)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			actual := getCgroupConfigInternal(tc.resourceList, tc.compressibleResources)
			tc.checks(actual, t)
		})
	}
}
