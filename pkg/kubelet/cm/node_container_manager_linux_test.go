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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/cpuset"

	libcontainercgroups "github.com/opencontainers/cgroups"
)

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

func TestSystemPartitionQOSContainersInfo(t *testing.T) {
	cases := []struct {
		name           string
		cgroupRoot     CgroupName
		wantGuaranteed CgroupName
		wantBurstable  CgroupName
		wantBestEffort CgroupName
	}{
		{
			name:           "default cgroup root",
			cgroupRoot:     NewCgroupName(RootCgroupName, defaultNodeAllocatableCgroupName),
			wantGuaranteed: NewCgroupName(RootCgroupName, "kubepods", "system"),
			wantBurstable:  NewCgroupName(RootCgroupName, "kubepods", "system", "burstable"),
			wantBestEffort: NewCgroupName(RootCgroupName, "kubepods", "system", "besteffort"),
		},
		{
			name:           "nested cgroup root",
			cgroupRoot:     NewCgroupName(RootCgroupName, "kubelet", defaultNodeAllocatableCgroupName),
			wantGuaranteed: NewCgroupName(RootCgroupName, "kubelet", "kubepods", "system"),
			wantBurstable:  NewCgroupName(RootCgroupName, "kubelet", "kubepods", "system", "burstable"),
			wantBestEffort: NewCgroupName(RootCgroupName, "kubelet", "kubepods", "system", "besteffort"),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := systemPartitionQOSContainersInfo(tc.cgroupRoot)
			// Guaranteed pods sit directly under the partition root, matching the
			// layout the QoS container manager creates under its own root.
			require.Equal(t, tc.wantGuaranteed, got.Guaranteed)
			require.Equal(t, tc.wantBurstable, got.Burstable)
			require.Equal(t, tc.wantBestEffort, got.BestEffort)
		})
	}
}

func TestSystemPartitionEnabled(t *testing.T) {
	withNamespaces := &SystemPartitionConfig{Namespaces: sets.New("kube-system")}
	withoutNamespaces := &SystemPartitionConfig{
		MemoryLimit: new(int64(1 << 30)),
		CPUSet:      cpuset.New(0, 1),
	}

	cases := []struct {
		name            string
		featureEnabled  bool
		cgroupsPerQOS   bool
		systemPartition *SystemPartitionConfig
		want            bool
	}{
		{
			name:            "all set",
			featureEnabled:  true,
			cgroupsPerQOS:   true,
			systemPartition: withNamespaces,
			want:            true,
		},
		{
			name:            "feature gate disabled",
			featureEnabled:  false,
			cgroupsPerQOS:   true,
			systemPartition: withNamespaces,
			want:            false,
		},
		{
			name:            "cgroupsPerQOS disabled",
			featureEnabled:  true,
			cgroupsPerQOS:   false,
			systemPartition: withNamespaces,
			want:            false,
		},
		{
			name:            "no system partition",
			featureEnabled:  true,
			cgroupsPerQOS:   true,
			systemPartition: nil,
			want:            false,
		},
		{
			name:            "system partition without namespaces",
			featureEnabled:  true,
			cgroupsPerQOS:   true,
			systemPartition: withoutNamespaces,
			want:            false,
		},
		{
			name:            "feature gate disabled and no system partition",
			featureEnabled:  false,
			cgroupsPerQOS:   true,
			systemPartition: nil,
			want:            false,
		},
		{
			name:            "cgroupsPerQOS disabled and no system partition",
			featureEnabled:  true,
			cgroupsPerQOS:   false,
			systemPartition: nil,
			want:            false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.NodeSystemPartition, tc.featureEnabled)

			nodeConfig := NodeConfig{
				CgroupsPerQOS:   tc.cgroupsPerQOS,
				SystemPartition: tc.systemPartition,
			}
			// The partition is scoped to cgroup v2, so nothing is enabled on a
			// cgroup v1 host regardless of the rest of the configuration.
			want := tc.want && libcontainercgroups.IsCgroup2UnifiedMode()
			require.Equal(t, want, systemPartitionEnabled(nodeConfig))
		})
	}
}

func TestSystemPartitionCgroupConfig(t *testing.T) {
	partitionRoot := NewCgroupName(RootCgroupName, defaultNodeAllocatableCgroupName, systemPartitionCgroupName)

	cases := []struct {
		name            string
		systemPartition *SystemPartitionConfig
		checks          func(*testing.T, *ResourceConfig)
	}{
		{
			name: "memory limit and cpuset",
			systemPartition: &SystemPartitionConfig{
				MemoryLimit: new(int64(4 << 30)),
				CPUSet:      cpuset.New(0, 1, 2, 3),
				Namespaces:  sets.New("kube-system"),
			},
			checks: func(t *testing.T, actual *ResourceConfig) {
				require.NotNil(t, actual.Memory)
				require.Equal(t, int64(4<<30), *actual.Memory)
				require.Equal(t, "0-3", actual.CPUSet.String())
			},
		},
		{
			name: "memory limit only",
			systemPartition: &SystemPartitionConfig{
				MemoryLimit: new(int64(4 << 30)),
				Namespaces:  sets.New("kube-system"),
			},
			checks: func(t *testing.T, actual *ResourceConfig) {
				require.NotNil(t, actual.Memory)
				require.Equal(t, int64(4<<30), *actual.Memory)
				require.True(t, actual.CPUSet.IsEmpty())
			},
		},
		{
			name: "cpuset only",
			systemPartition: &SystemPartitionConfig{
				CPUSet:     cpuset.New(0, 1),
				Namespaces: sets.New("kube-system"),
			},
			checks: func(t *testing.T, actual *ResourceConfig) {
				require.Nil(t, actual.Memory)
				require.Equal(t, "0-1", actual.CPUSet.String())
			},
		},
		{
			// A partition with neither limit still moves its pods into their own
			// QoS hierarchy, so the cgroup is created with no limits on it.
			name: "no limits",
			systemPartition: &SystemPartitionConfig{
				Namespaces: sets.New("kube-system"),
			},
			checks: func(t *testing.T, actual *ResourceConfig) {
				require.Nil(t, actual.Memory)
				require.True(t, actual.CPUSet.IsEmpty())
				require.Nil(t, actual.CPUShares)
				require.Nil(t, actual.CPUQuota)
				require.Nil(t, actual.PidsLimit)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cm := &containerManagerImpl{
				NodeConfig:          NodeConfig{SystemPartition: tc.systemPartition},
				systemPartitionRoot: partitionRoot,
			}
			got := cm.systemPartitionCgroupConfig()
			require.Equal(t, partitionRoot, got.Name)
			// Nothing is derived from node capacity, unlike the node allocatable cgroup.
			tc.checks(t, got.ResourceParameters)
		})
	}
}

func TestSystemPartitionPodsAndAllocatable(t *testing.T) {
	t.Run("without a system partition every pod is outside it", func(t *testing.T) {
		pods := []*v1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "a", Namespace: "kube-system"}}}
		cm := &containerManagerImpl{}
		require.Equal(t, pods, cm.podsInSystemPartition(func() []*v1.Pod { return pods }, false)())
		require.Empty(t, cm.podsInSystemPartition(func() []*v1.Pod { return pods }, true)())
	})

	inPartition := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "in", Namespace: "kube-system"}}
	outside := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "out", Namespace: metav1.NamespaceDefault}}
	activePods := func() []*v1.Pod { return []*v1.Pod{inPartition, outside} }
	capacity := v1.ResourceList{
		v1.ResourceCPU:    resource.MustParse("8"),
		v1.ResourceMemory: resource.MustParse("16Gi"),
	}

	cases := []struct {
		name        string
		memoryLimit *int64
		wantMemory  string
	}{
		{
			name:        "the memory limit replaces the node's allocatable memory",
			memoryLimit: new(int64(1 << 30)),
			wantMemory:  "1Gi",
		},
		{
			// Nothing bounds the partition's pods but the node then.
			name:       "without a memory limit the node's allocatable memory is kept",
			wantMemory: "16Gi",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cm := &containerManagerImpl{
				capacity: capacity,
				NodeConfig: NodeConfig{SystemPartition: &SystemPartitionConfig{
					MemoryLimit: tc.memoryLimit,
					Namespaces:  sets.New("kube-system"),
				}},
				systemPartitionQOSManager: &qosContainerManagerNoop{},
			}

			require.Equal(t, []*v1.Pod{inPartition}, cm.podsInSystemPartition(activePods, true)())
			require.Equal(t, []*v1.Pod{outside}, cm.podsInSystemPartition(activePods, false)())

			allocatable := cm.systemPartitionAllocatable()
			memory := allocatable[v1.ResourceMemory]
			wantMemory := resource.MustParse(tc.wantMemory)
			require.Equal(t, wantMemory.Value(), memory.Value())
			cpu := allocatable[v1.ResourceCPU]
			require.Equal(t, int64(8), cpu.Value(), "only memory is bounded by the partition")
			// The node's own allocatable must not be changed by the partition's.
			nodeMemory := cm.GetNodeAllocatableAbsolute()[v1.ResourceMemory]
			wantNodeMemory := capacity[v1.ResourceMemory]
			require.Equal(t, wantNodeMemory.Value(), nodeMemory.Value())
		})
	}
}

func TestUpdateSystemPartitionCPUWeight(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	partitionRoot := NewCgroupName(RootCgroupName, defaultNodeAllocatableCgroupName, systemPartitionCgroupName)
	// The node's pods as the kubelet lists them, both partitions mixed.
	nodePods := []*v1.Pod{
		newPartitionTestPod("kube-system", "200m/128Mi", "200m/128Mi"),
		newPartitionTestPod(metav1.NamespaceDefault, "1/1Gi", "1/1Gi"),
		newPartitionTestPod("kube-system", "100m/64Mi", "200m/128Mi"),
		newPartitionTestPod("kube-system", "", ""),
	}
	// A pod that sets its requests at the pod level only.
	podLevelPod := newPartitionTestPod("kube-system", "", "")
	podLevelPod.Spec.Resources = &v1.ResourceRequirements{
		Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("300m"), v1.ResourceMemory: resource.MustParse("128Mi")},
		Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("300m"), v1.ResourceMemory: resource.MustParse("128Mi")},
	}

	cases := []struct {
		name              string
		pods              []*v1.Pod
		podLevelResources bool
		noPods            bool
		wantShares        *uint64
	}{
		{
			name:       "the partition is weighted by the requests of its own pods",
			pods:       nodePods,
			wantShares: new(MilliCPUToShares(300)),
		},
		{
			name:              "pod-level requests weight the partition",
			pods:              append([]*v1.Pod{podLevelPod}, nodePods...),
			podLevelResources: true,
			wantShares:        new(MilliCPUToShares(600)),
		},
		{
			name:       "an empty partition gets the minimum weight, not the cgroup default",
			wantShares: new(uint64(MinShares)),
		},
		{
			// Before the partition's QoS cgroups are set up.
			name:   "nothing is written without the partition's pods",
			noPods: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.PodLevelResources, tc.podLevelResources)
			fakeCM := &fakeCgroupManager{}
			cm := &containerManagerImpl{
				cgroupManager:             fakeCM,
				systemPartitionRoot:       partitionRoot,
				NodeConfig:                NodeConfig{SystemPartition: &SystemPartitionConfig{Namespaces: sets.New("kube-system")}},
				systemPartitionQOSManager: &qosContainerManagerNoop{},
			}
			if !tc.noPods {
				cm.systemPartitionPods = cm.podsInSystemPartition(func() []*v1.Pod { return tc.pods }, true)
			}

			require.NoError(t, cm.updateSystemPartitionCPUWeight(logger))

			if tc.wantShares == nil {
				require.Empty(t, fakeCM.updates)
				return
			}
			require.Len(t, fakeCM.updates, 1)
			require.Equal(t, partitionRoot, fakeCM.updates[0].Name)
			require.Equal(t, *tc.wantShares, *fakeCM.updates[0].ResourceParameters.CPUShares)
			require.Nil(t, fakeCM.updates[0].ResourceParameters.Memory, "only the weight is written, the memory limit is left alone")
		})
	}
}
