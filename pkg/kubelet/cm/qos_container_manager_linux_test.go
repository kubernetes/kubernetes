//go:build linux

/*
Copyright 2021 The Kubernetes Authors.

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
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	resource "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
)

func activeTestPods() []*v1.Pod {
	return []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:       "12345678",
				Name:      "guaranteed-pod",
				Namespace: "test",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "foo",
						Image: "busybox",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("128Mi"),
								v1.ResourceCPU:    resource.MustParse("1"),
							},
							Limits: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("128Mi"),
								v1.ResourceCPU:    resource.MustParse("1"),
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:       "87654321",
				Name:      "burstable-pod-1",
				Namespace: "test",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "foo",
						Image: "busybox",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("128Mi"),
								v1.ResourceCPU:    resource.MustParse("1"),
							},
							Limits: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("256Mi"),
								v1.ResourceCPU:    resource.MustParse("2"),
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:       "01234567",
				Name:      "burstable-pod-2",
				Namespace: "test",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "foo",
						Image: "busybox",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("256Mi"),
								v1.ResourceCPU:    resource.MustParse("2"),
							},
						},
					},
				},
			},
		},
	}
}

func createTestQOSContainerManager(logger klog.Logger) (*qosContainerManagerImpl, error) {
	subsystems, err := GetCgroupSubsystems()
	if err != nil {
		return nil, fmt.Errorf("failed to get mounted cgroup subsystems: %v", err)
	}

	cgroupRoot := ParseCgroupfsToCgroupName("/")
	cgroupRoot = NewCgroupName(cgroupRoot, defaultNodeAllocatableCgroupName)

	qosContainerManager := &qosContainerManagerImpl{
		subsystems:              subsystems,
		cgroupManager:           NewCgroupManager(logger, subsystems, "cgroupfs"),
		cgroupRoot:              cgroupRoot,
		qosReserved:             nil,
		memoryReservationPolicy: kubeletconfig.NoneMemoryReservationPolicy,
	}

	qosContainerManager.activePods = activeTestPods

	return qosContainerManager, nil
}

func TestQoSContainerCgroup(t *testing.T) {
	burstableMin := resource.MustParse("384Mi")
	guaranteedMin := resource.MustParse("128Mi")

	tests := []struct {
		name                               string
		pods                               []*v1.Pod
		initialGuaranteed                  string
		initialBurstable                   string
		expectedGuaranteed                 string
		expectedBurstable                  string
		draNodeAllocatableResourcesEnabled bool
		podLevelResourcesEnabled           bool
	}{
		{
			name:               "writes aggregated memory min",
			pods:               activeTestPods(),
			initialGuaranteed:  "",
			initialBurstable:   "",
			expectedGuaranteed: strconv.FormatInt(burstableMin.Value()+guaranteedMin.Value(), 10),
			expectedBurstable:  strconv.FormatInt(burstableMin.Value(), 10),
		},
		{
			name: "writes zero memory min for best effort pod",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{UID: "99999999", Name: "besteffort-pod", Namespace: "test"},
					Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "foo", Image: "busybox"}}},
				},
			},
			initialGuaranteed:  "",
			initialBurstable:   "",
			expectedGuaranteed: "0",
			expectedBurstable:  "0",
		},
		{
			name: "writes zero memory min for burstable pod without memory request",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{UID: "88888888", Name: "burstable-pod-no-memory-request", Namespace: "test"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name:  "foo",
								Image: "busybox",
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
								},
							},
						},
					},
				},
			},
			initialGuaranteed:  "",
			initialBurstable:   "",
			expectedGuaranteed: "0",
			expectedBurstable:  "0",
		},
		{
			name:               "clears stale memory min when all pods removed",
			pods:               nil,
			initialGuaranteed:  "1234",
			initialBurstable:   "5678",
			expectedGuaranteed: "0",
			expectedBurstable:  "0",
		},
		{
			name: "writes zero memory min for best effort pod with DRA",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{UID: "99999999", Name: "besteffort-pod-with-dra", Namespace: "test"},
					Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "foo", Image: "busybox"}}},
					Status: v1.PodStatus{
						NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
							{
								ResourceClaimName: "direct-claim",
								Containers:        []string{"foo"},
								Mapping: []v1.NodeAllocatableMappedResources{
									{Name: v1.ResourceMemory, Quantity: new(resource.MustParse("128Mi"))},
								},
							},
						},
					},
				},
			},
			draNodeAllocatableResourcesEnabled: true,
			initialGuaranteed:                  "",
			initialBurstable:                   "",
			expectedGuaranteed:                 "0",
			expectedBurstable:                  "0",
		},
		{
			name: "writes memory min including DRA allocations",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{UID: "12345678", Name: "guaranteed-pod", Namespace: "test"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name:  "foo",
								Image: "busybox",
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceMemory: resource.MustParse("128Mi"),
										v1.ResourceCPU:    resource.MustParse("1"),
									},
									Limits: v1.ResourceList{
										v1.ResourceMemory: resource.MustParse("128Mi"),
										v1.ResourceCPU:    resource.MustParse("1"),
									},
								},
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{UID: "87654321", Name: "burstable-pod", Namespace: "test"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name:  "foo",
								Image: "busybox",
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceMemory: resource.MustParse("128Mi"),
										v1.ResourceCPU:    resource.MustParse("1"),
									},
									Limits: v1.ResourceList{
										v1.ResourceMemory: resource.MustParse("256Mi"),
										v1.ResourceCPU:    resource.MustParse("2"),
									},
								},
							},
						},
					},
					Status: v1.PodStatus{
						NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
							{
								ResourceClaimName: "direct-claim",
								Containers:        []string{"foo"},
								Mapping: []v1.NodeAllocatableMappedResources{
									{Name: v1.ResourceMemory, Quantity: new(resource.MustParse("128Mi"))},
								},
							},
						},
					},
				},
			},
			draNodeAllocatableResourcesEnabled: true,
			initialGuaranteed:                  "",
			initialBurstable:                   "",
			expectedGuaranteed:                 strconv.FormatInt(384*1024*1024, 10), // Guaranteed 128Mi + Burstable (128Mi spec + 128Mi DRA) = 384Mi
			expectedBurstable:                  strconv.FormatInt(256*1024*1024, 10), // Burstable 128Mi spec + 128Mi DRA = 256Mi
		},
		{
			name: "writes memory min with pod-level resources",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{UID: "12345678", Name: "guaranteed-pod", Namespace: "test"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name:  "foo",
								Image: "busybox",
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceMemory: resource.MustParse("128Mi"),
										v1.ResourceCPU:    resource.MustParse("1"),
									},
									Limits: v1.ResourceList{
										v1.ResourceMemory: resource.MustParse("128Mi"),
										v1.ResourceCPU:    resource.MustParse("1"),
									},
								},
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{UID: "87654321", Name: "burstable-pod-pod-level", Namespace: "test"},
					Spec: v1.PodSpec{
						Resources: &v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("512Mi"),
							},
						},
						Containers: []v1.Container{
							{
								Name:  "foo",
								Image: "busybox",
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceMemory: resource.MustParse("128Mi"),
										v1.ResourceCPU:    resource.MustParse("1"),
									},
									Limits: v1.ResourceList{
										v1.ResourceMemory: resource.MustParse("256Mi"),
										v1.ResourceCPU:    resource.MustParse("2"),
									},
								},
							},
						},
					},
					Status: v1.PodStatus{
						NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
							{
								ResourceClaimName: "direct-claim",
								Containers:        []string{"foo"},
								Mapping: []v1.NodeAllocatableMappedResources{
									{Name: v1.ResourceMemory, Quantity: new(resource.MustParse("128Mi"))},
								},
							},
						},
					},
				},
			},
			draNodeAllocatableResourcesEnabled: true,
			podLevelResourcesEnabled:           true,
			initialGuaranteed:                  "",
			initialBurstable:                   "",
			expectedGuaranteed:                 strconv.FormatInt(640*1024*1024, 10), // Guaranteed 128Mi + Burstable pod-level 512Mi = 640Mi
			expectedBurstable:                  strconv.FormatInt(512*1024*1024, 10), // Burstable pod-level 512Mi
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.DRANodeAllocatableResources, tc.draNodeAllocatableResourcesEnabled)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, tc.podLevelResourcesEnabled)
			logger, _ := ktesting.NewTestContext(t)
			m, err := createTestQOSContainerManager(logger)
			require.NoError(t, err)
			// Set memory reservation policy to TieredReservation to enable memory.min
			m.memoryReservationPolicy = kubeletconfig.TieredReservationMemoryReservationPolicy
			m.activePods = func() []*v1.Pod { return tc.pods }

			guaranteedUnified := map[string]string{}
			if tc.initialGuaranteed != "" {
				guaranteedUnified[Cgroup2MemoryMin] = tc.initialGuaranteed
			}
			burstableUnified := map[string]string{}
			if tc.initialBurstable != "" {
				burstableUnified[Cgroup2MemoryLow] = tc.initialBurstable
			}

			qosConfigs := map[v1.PodQOSClass]*CgroupConfig{
				v1.PodQOSGuaranteed: {
					Name: m.qosContainersInfo.Guaranteed,
					ResourceParameters: &ResourceConfig{
						Unified: guaranteedUnified,
					},
				},
				v1.PodQOSBurstable: {
					Name: m.qosContainersInfo.Burstable,
					ResourceParameters: &ResourceConfig{
						Unified: burstableUnified,
					},
				},
				v1.PodQOSBestEffort: {
					Name:               m.qosContainersInfo.BestEffort,
					ResourceParameters: &ResourceConfig{},
				},
			}

			m.setMemoryQoS(logger, qosConfigs)

			assert.Equal(t, tc.expectedGuaranteed, qosConfigs[v1.PodQOSGuaranteed].ResourceParameters.Unified[Cgroup2MemoryMin])
			assert.Equal(t, tc.expectedBurstable, qosConfigs[v1.PodQOSBurstable].ResourceParameters.Unified[Cgroup2MemoryLow])
		})
	}
}

func TestQoSContainerCgroupWithMemoryReservationPolicyNone(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	fakeCM := &fakeCgroupManager{}
	cgroupRoot := ParseCgroupfsToCgroupName("/")
	cgroupRoot = NewCgroupName(cgroupRoot, defaultNodeAllocatableCgroupName)
	m := &qosContainerManagerImpl{
		cgroupManager:           fakeCM,
		cgroupRoot:              cgroupRoot,
		activePods:              activeTestPods,
		memoryReservationPolicy: kubeletconfig.NoneMemoryReservationPolicy,
		qosContainersInfo: QOSContainersInfo{
			Guaranteed: cgroupRoot,
			Burstable:  NewCgroupName(cgroupRoot, "burstable"),
			BestEffort: NewCgroupName(cgroupRoot, "besteffort"),
		},
	}

	// memoryReservationPolicy defaults to NoneMemoryReservationPolicy, so memory.min should be explicitly reset.
	qosConfigs := map[v1.PodQOSClass]*CgroupConfig{
		v1.PodQOSGuaranteed: {
			Name:               m.qosContainersInfo.Guaranteed,
			ResourceParameters: &ResourceConfig{},
		},
		v1.PodQOSBurstable: {
			Name:               m.qosContainersInfo.Burstable,
			ResourceParameters: &ResourceConfig{},
		},
		v1.PodQOSBestEffort: {
			Name:               m.qosContainersInfo.BestEffort,
			ResourceParameters: &ResourceConfig{},
		},
	}

	m.setMemoryQoS(logger, qosConfigs)

	assert.Equal(t, "0", qosConfigs[v1.PodQOSGuaranteed].ResourceParameters.Unified[Cgroup2MemoryMin])
	assert.Equal(t, "0", qosConfigs[v1.PodQOSBurstable].ResourceParameters.Unified[Cgroup2MemoryLow])
}

// fakeCgroupManager is used because Start() requires a functional
// CgroupManager. All methods are stubbed so that Start() can
// complete successfully without using real cgroups.
type fakeCgroupManager struct {
	mutex   sync.Mutex
	created []*CgroupConfig
	updates []*CgroupConfig
}

// Update() is the observation point for this test.
// Capture the updated cgroup config so it can be validated.
func (f *fakeCgroupManager) Update(_ klog.Logger, config *CgroupConfig) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	copiedConfig := *config
	f.updates = append(f.updates, &copiedConfig)
	return nil
}

// Create() must succeed for Start() to construct QoS cgroups.
// We do not assert on Create() behavior in this test.
func (f *fakeCgroupManager) Create(_ klog.Logger, config *CgroupConfig) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	copiedConfig := *config
	f.created = append(f.created, &copiedConfig)
	return nil
}

func (f *fakeCgroupManager) Destroy(_ klog.Logger, _ *CgroupConfig) error { return nil }
func (f *fakeCgroupManager) Validate(_ CgroupName) error                  { return nil }
func (f *fakeCgroupManager) Exists(_ CgroupName) bool                     { return false }
func (f *fakeCgroupManager) Name(name CgroupName) string                  { return name.ToCgroupfs() }
func (f *fakeCgroupManager) CgroupName(name string) CgroupName {
	return ParseCgroupfsToCgroupName(name)
}
func (f *fakeCgroupManager) Pids(_ klog.Logger, _ CgroupName) []int { return nil }
func (f *fakeCgroupManager) ReduceCPULimits(_ klog.Logger, _ CgroupName) error {
	return nil
}
func (f *fakeCgroupManager) MemoryUsage(_ CgroupName) (int64, error) { return int64(0), nil }
func (f *fakeCgroupManager) GetCgroupConfig(_ CgroupName, _ v1.ResourceName) (*ResourceConfig, error) {
	return nil, nil
}
func (f *fakeCgroupManager) SetCgroupConfig(_ klog.Logger, _ CgroupName, _ *ResourceConfig) error {
	return nil
}
func (f *fakeCgroupManager) Version() int { return 1 }

// TestQOSCPUConfigUpdate verifies that UpdateCgroups() computes and
// updates the correct CPU shares for each QoS class based on the
// currently active pods.
func TestQOSCPUConfigUpdate(t *testing.T) {

	// Guaranteed QoS uses fixed CPU shares (requests == limits), so they are not
	// recalculated. BestEffort always uses MinShares, and only Burstable CPU shares
	// depend on aggregate burstable CPU requests.

	tests := []struct {
		name                       string
		testPods                   ActivePodsFunc
		expectedBurstableCPUShares uint64 // Recalculation will be done only for Burstable QoS class
	}{
		{
			name: "guaranteed-pods-only",
			testPods: func() []*v1.Pod {
				return []*v1.Pod{

					{
						ObjectMeta: metav1.ObjectMeta{
							UID:       types.UID(uuid.NewUUID()),
							Name:      "guaranteed-pod",
							Namespace: "test",
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "foo",
									Image: "busybox",
									Resources: v1.ResourceRequirements{
										Requests: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("1"),
											v1.ResourceMemory: resource.MustParse("128Mi"),
										},
										Limits: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("1"),
											v1.ResourceMemory: resource.MustParse("128Mi"),
										},
									},
								},
							},
						},
					},
				}
			},
			// MinShares will be given to the Burstable QoS class since kubelet
			// creates all QoS cgroups regardless of whether pods of that class exist.
			expectedBurstableCPUShares: MinShares,
		},
		{
			name: "burstable-pods-only",
			testPods: func() []*v1.Pod {
				return []*v1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							UID:       types.UID(uuid.NewUUID()),
							Name:      "burstable-pod",
							Namespace: "test",
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "foo",
									Image: "busybox",
									Resources: v1.ResourceRequirements{
										Requests: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("1"),
											v1.ResourceMemory: resource.MustParse("128Mi"),
										},
										Limits: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("2"),
											v1.ResourceMemory: resource.MustParse("256Mi"),
										},
									},
								},
							},
						},
					},
				}
			},
			// 1 CPU Resource = 1024 CPU Shares
			expectedBurstableCPUShares: 1024,
		},
		{
			name: "besteffort-pods-only",
			testPods: func() []*v1.Pod {
				return []*v1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							UID:       types.UID(uuid.NewUUID()),
							Name:      "besteffort-pod",
							Namespace: "test",
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "foo",
									Image: "busybox",
								},
							},
						},
					},
				}
			},
			expectedBurstableCPUShares: MinShares,
		},
		{
			name: "guaranteed-and-burstable-pods",
			testPods: func() []*v1.Pod {
				return []*v1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							UID:       types.UID(uuid.NewUUID()),
							Name:      "guaranteed-pod",
							Namespace: "test",
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "foo",
									Image: "busybox",
									Resources: v1.ResourceRequirements{
										Requests: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("1"),
											v1.ResourceMemory: resource.MustParse("128Mi"),
										},
										Limits: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("1"),
											v1.ResourceMemory: resource.MustParse("128Mi"),
										},
									},
								},
							},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							UID:       types.UID(uuid.NewUUID()),
							Name:      "burstable-pod",
							Namespace: "test",
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "foo",
									Image: "busybox",
									Resources: v1.ResourceRequirements{
										Requests: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("1"),
											v1.ResourceMemory: resource.MustParse("128Mi"),
										},
										Limits: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("2"),
											v1.ResourceMemory: resource.MustParse("256Mi"),
										},
									},
								},
							},
						},
					},
				}
			},
			expectedBurstableCPUShares: 1024,
		},
		{
			name: "besteffort-and-burstable-pods",
			testPods: func() []*v1.Pod {
				return []*v1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							UID:       types.UID(uuid.NewUUID()),
							Name:      "besteffort-pod",
							Namespace: "test",
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "foo",
									Image: "busybox",
								},
							},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							UID:       types.UID(uuid.NewUUID()),
							Name:      "burstable-pod",
							Namespace: "test",
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "foo",
									Image: "busybox",
									Resources: v1.ResourceRequirements{
										Requests: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("1"),
											v1.ResourceMemory: resource.MustParse("128Mi"),
										},
										Limits: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("2"),
											v1.ResourceMemory: resource.MustParse("256Mi"),
										},
									},
								},
							},
						},
					},
				}
			},
			expectedBurstableCPUShares: 1024,
		},
	}

	for _, testCase := range tests {

		t.Run(testCase.name, func(t *testing.T) {

			logger, ctx := ktesting.NewTestContext(t)

			testContainerManager, err := createTestQOSContainerManager(logger)
			if err != nil {
				t.Fatalf("Unable to create Test Qos Container Manager: %s", err)
				return
			}

			fakecgroupManager := &fakeCgroupManager{}
			testContainerManager.cgroupManager = fakecgroupManager

			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			err = testContainerManager.Start(ctx, func() v1.ResourceList { return v1.ResourceList{} }, testCase.testPods, nil)

			if err != nil {
				t.Fatalf("Start() failed: %s", err)
			}

			// UpdateCgroups() is expected to update all QoS cgroups on each call
			// based on the current active pod set.
			err = testContainerManager.UpdateCgroups(logger)
			if err != nil {
				t.Fatalf("Error in UpdateCgroups(): %s", err)
			}
			cancel()

			// UpdateCgroups() may also be running in the background reconciliation loop (because of Start()).
			// Retry until a consistent snapshot containing all QoS cgroup updates is observed.
			//
			// A small bounded retry count is sufficient here because UpdateCgroups()
			// is synchronous and expected to complete quickly. This avoids test flakiness
			// without risking an unbounded wait.
			maxRetryAttempts := 5

			var (
				foundBurstable  bool
				foundBestEffort bool
				foundGuaranteed bool
			)

			for i := 0; i < maxRetryAttempts; i++ {

				// These flags will be used to check whether UpdateCgroups()
				// is updating the CPU shares for all QoS classes (cgroups)
				foundBurstable = false
				foundBestEffort = false
				foundGuaranteed = false

				// Start() initiates a background UpdateCgroups() goroutine which may
				// still be running. Take a snapshot to avoid observing updates from
				// the background UpdateCgroups() instead of the explicit UpdateCgroups() call.
				fakecgroupManager.mutex.Lock()
				updates := append([]*CgroupConfig(nil), fakecgroupManager.updates...)
				fakecgroupManager.mutex.Unlock()

				for _, config := range updates {

					if strings.HasSuffix(config.Name.ToCgroupfs(), "burstable") {
						foundBurstable = true
						if *config.ResourceParameters.CPUShares != testCase.expectedBurstableCPUShares {
							t.Fatalf("Expected CPU Shares for Burstable: %d Got: %d", testCase.expectedBurstableCPUShares, *config.ResourceParameters.CPUShares)
						}
						continue
					}

					if strings.HasSuffix(config.Name.ToCgroupfs(), "besteffort") {
						foundBestEffort = true
						if *config.ResourceParameters.CPUShares != MinShares {
							t.Fatalf("Expected CPU Shares for BestEffort: %d Got: %d", MinShares, *config.ResourceParameters.CPUShares)
						}
						continue
					}

					if config.Name.ToCgroupfs() == testContainerManager.cgroupRoot.ToCgroupfs() {
						foundGuaranteed = true
						if config.ResourceParameters != nil && config.ResourceParameters.CPUShares != nil {
							t.Fatalf("Expected CPU Shares for Guaranteed: <nil>, Got: %d", *config.ResourceParameters.CPUShares)
						}
					}
				}

				if foundBurstable && foundBestEffort && foundGuaranteed {
					break
				}

				time.Sleep(time.Millisecond * 10)
			}

			if !foundBurstable || !foundBestEffort || !foundGuaranteed {

				t.Fatalf(
					"did not observe all QoS cgroup updates after %d retries (guaranteed=%v, burstable=%v, besteffort=%v)",
					maxRetryAttempts, foundGuaranteed, foundBurstable, foundBestEffort,
				)
			}
		})
	}
}

// newPartitionTestPod returns a pod with a single container of the given
// requests and limits, in "cpu/memory" form; an empty string leaves both unset.
func newPartitionTestPod(namespace, requests, limits string) *v1.Pod {
	toList := func(s string) v1.ResourceList {
		if s == "" {
			return nil
		}
		cpu, memory, _ := strings.Cut(s, "/")
		return v1.ResourceList{v1.ResourceCPU: resource.MustParse(cpu), v1.ResourceMemory: resource.MustParse(memory)}
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: string(uuid.NewUUID()), Namespace: namespace, UID: uuid.NewUUID()},
		Spec: v1.PodSpec{Containers: []v1.Container{{
			Name:      "c",
			Resources: v1.ResourceRequirements{Requests: toList(requests), Limits: toList(limits)},
		}}},
	}
}

// qosValues are the values a QoS manager computes for one QoS cgroup. A nil
// field is left unset.
type qosValues struct {
	CPUShares *uint64
	MemoryMin *int64
	MemoryLow *int64
	MemoryMax *int64
}

// toQOSValues extracts the values that the QoS manager put in configs.
func toQOSValues(t *testing.T, configs map[v1.PodQOSClass]*CgroupConfig) map[v1.PodQOSClass]qosValues {
	t.Helper()
	unified := func(config *CgroupConfig, key string) *int64 {
		value, found := config.ResourceParameters.Unified[key]
		if !found {
			return nil
		}
		parsed, err := strconv.ParseInt(value, 10, 64)
		require.NoError(t, err, "%s of %s", key, config.Name)
		return &parsed
	}
	values := map[v1.PodQOSClass]qosValues{}
	for qosClass, config := range configs {
		values[qosClass] = qosValues{
			CPUShares: config.ResourceParameters.CPUShares,
			MemoryMin: unified(config, Cgroup2MemoryMin),
			MemoryLow: unified(config, Cgroup2MemoryLow),
			MemoryMax: config.ResourceParameters.Memory,
		}
	}
	return values
}

func TestQOSCgroupsWithSystemPartition(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.MemoryQoS, true)
	kubeletmetrics.Register()
	// The gauges are global, so leave them as other tests expect to find them.
	t.Cleanup(func() {
		kubeletmetrics.MemoryQoSNodeMemoryMinBytes.Set(0)
		kubeletmetrics.MemoryQoSNodeMemoryLowBytes.Set(0)
	})

	const (
		systemNamespace = "kube-system"
		mi              = int64(1 << 20)

		// The requests of the node's pods. A burstable pod's limits are twice
		// its requests.
		defaultGuaranteedCPU    = int64(1000)
		defaultGuaranteedMemory = 1024 * mi
		defaultBurstableCPU     = int64(500)
		defaultBurstableMemory  = 256 * mi
		systemGuaranteedCPU     = int64(200)
		systemGuaranteedMemory  = 128 * mi
		systemBurstableCPU      = int64(100)
		systemBurstableMemory   = 64 * mi

		nodeAllocatableMemory = 8192 * mi
		partitionMemoryLimit  = 1024 * mi
		percentReserve        = 100
	)
	guaranteedPod := func(namespace string, cpu, memory int64) *v1.Pod {
		requests := fmt.Sprintf("%dm/%d", cpu, memory)
		return newPartitionTestPod(namespace, requests, requests)
	}
	burstablePod := func(namespace string, cpu, memory int64) *v1.Pod {
		return newPartitionTestPod(namespace, fmt.Sprintf("%dm/%d", cpu, memory), fmt.Sprintf("%dm/%d", 2*cpu, 2*memory))
	}
	// The node's pods as the kubelet lists them, both partitions mixed.
	nodePods := []*v1.Pod{
		guaranteedPod(metav1.NamespaceDefault, defaultGuaranteedCPU, defaultGuaranteedMemory),
		guaranteedPod(systemNamespace, systemGuaranteedCPU, systemGuaranteedMemory),
		burstablePod(metav1.NamespaceDefault, defaultBurstableCPU, defaultBurstableMemory),
		burstablePod(systemNamespace, systemBurstableCPU, systemBurstableMemory),
		newPartitionTestPod(metav1.NamespaceDefault, "", ""),
		newPartitionTestPod(systemNamespace, "", ""),
	}
	activePods := func() []*v1.Pod { return nodePods }

	nodeConfig := NodeConfig{
		CgroupsPerQOS:           true,
		MemoryReservationPolicy: kubeletconfig.TieredReservationMemoryReservationPolicy,
	}
	nodeRoot := NewCgroupName(RootCgroupName, defaultNodeAllocatableCgroupName)
	partitionRoot := NewCgroupName(nodeRoot, systemPartitionCgroupName)
	cm := &containerManagerImpl{
		NodeConfig:                NodeConfig{SystemPartition: &SystemPartitionConfig{Namespaces: sets.New(systemNamespace)}},
		systemPartitionQOSManager: &qosContainerManagerNoop{},
	}

	type nodeMetrics struct {
		MemoryMin float64
		MemoryLow float64
	}
	// Preset before each case, so that a manager that must not report them
	// can be told apart from one that reports zero.
	presetMetrics := nodeMetrics{MemoryMin: 1, MemoryLow: 2}

	// The node's pods sit in two hierarchies, each written by its own QoS
	// manager:
	//
	//   kubepods             [node]       protects every pod below
	//   ├── burstable        [node]
	//   ├── besteffort       [node]
	//   ├── pod
	//   └── system           [partition]  protects the partition's pods
	//       ├── burstable    [partition]
	//       ├── besteffort   [partition]
	//       └── pod
	//
	// Each QoS cgroup counts only the pods in it, while a root counts every pod
	// below it, since a pod's memory protection is bounded by its ancestors'.
	// The burstable cgroup still leaves room for every other pod under the root.
	cases := []struct {
		name            string
		root            CgroupName
		skipNodeMetrics bool
		allocatable     int64
		activePods      ActivePodsFunc
		podsUnderRoot   ActivePodsFunc
		want            map[v1.PodQOSClass]qosValues
		wantMetrics     nodeMetrics
	}{
		{
			name:          "the default partition sizes its QoS cgroups by its own pods and its root by all pods",
			root:          nodeRoot,
			allocatable:   nodeAllocatableMemory,
			activePods:    cm.podsInSystemPartition(activePods, false),
			podsUnderRoot: activePods,
			want: map[v1.PodQOSClass]qosValues{
				v1.PodQOSGuaranteed: {
					MemoryMin: new(defaultGuaranteedMemory + defaultBurstableMemory + systemGuaranteedMemory + systemBurstableMemory),
					MemoryLow: new(defaultBurstableMemory + systemBurstableMemory),
				},
				v1.PodQOSBurstable: {
					CPUShares: new(MilliCPUToShares(defaultBurstableCPU)),
					MemoryLow: new(defaultBurstableMemory),
					MemoryMax: new(nodeAllocatableMemory - (defaultGuaranteedMemory + systemGuaranteedMemory + systemBurstableMemory)),
				},
				v1.PodQOSBestEffort: {
					CPUShares: new(uint64(MinShares)),
					MemoryMax: new(nodeAllocatableMemory - (defaultGuaranteedMemory + systemGuaranteedMemory + systemBurstableMemory) - defaultBurstableMemory),
				},
			},
			wantMetrics: nodeMetrics{
				MemoryMin: float64(defaultGuaranteedMemory + systemGuaranteedMemory),
				MemoryLow: float64(defaultBurstableMemory + systemBurstableMemory),
			},
		},
		{
			name:            "the system partition sizes its QoS cgroups by its own pods and memory limit",
			root:            partitionRoot,
			skipNodeMetrics: true,
			allocatable:     partitionMemoryLimit,
			activePods:      cm.podsInSystemPartition(activePods, true),
			want: map[v1.PodQOSClass]qosValues{
				// The partition root's CPU weight is the container manager's
				// to set, so it is left alone here.
				v1.PodQOSGuaranteed: {
					MemoryMin: new(systemGuaranteedMemory + systemBurstableMemory),
					MemoryLow: new(systemBurstableMemory),
				},
				v1.PodQOSBurstable: {
					CPUShares: new(MilliCPUToShares(systemBurstableCPU)),
					MemoryLow: new(systemBurstableMemory),
					MemoryMax: new(partitionMemoryLimit - systemGuaranteedMemory),
				},
				v1.PodQOSBestEffort: {
					CPUShares: new(uint64(MinShares)),
					MemoryMax: new(partitionMemoryLimit - systemGuaranteedMemory - systemBurstableMemory),
				},
			},
			// A partition only sees part of the node, so it must not report
			// node-wide metrics.
			wantMetrics: presetMetrics,
		},
		{
			// The [node] manager as the kubelet builds it without a partition:
			// kubepods has no system child, so the root and its QoS cgroups count
			// the same pods.
			name:        "a node without a partition reports the node-wide metrics",
			root:        nodeRoot,
			allocatable: nodeAllocatableMemory,
			activePods:  activePods,
			want: map[v1.PodQOSClass]qosValues{
				v1.PodQOSGuaranteed: {
					MemoryMin: new(defaultGuaranteedMemory + defaultBurstableMemory + systemGuaranteedMemory + systemBurstableMemory),
					MemoryLow: new(defaultBurstableMemory + systemBurstableMemory),
				},
				v1.PodQOSBurstable: {
					CPUShares: new(MilliCPUToShares(defaultBurstableCPU + systemBurstableCPU)),
					MemoryLow: new(defaultBurstableMemory + systemBurstableMemory),
					MemoryMax: new(nodeAllocatableMemory - (defaultGuaranteedMemory + systemGuaranteedMemory)),
				},
				v1.PodQOSBestEffort: {
					CPUShares: new(uint64(MinShares)),
					MemoryMax: new(nodeAllocatableMemory - (defaultGuaranteedMemory + systemGuaranteedMemory) - (defaultBurstableMemory + systemBurstableMemory)),
				},
			},
			wantMetrics: nodeMetrics{
				MemoryMin: float64(defaultGuaranteedMemory + systemGuaranteedMemory),
				MemoryLow: float64(defaultBurstableMemory + systemBurstableMemory),
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			kubeletmetrics.MemoryQoSNodeMemoryMinBytes.Set(presetMetrics.MemoryMin)
			kubeletmetrics.MemoryQoSNodeMemoryLowBytes.Set(presetMetrics.MemoryLow)

			m := newQOSContainerManager(nil, tc.root, nodeConfig, &fakeCgroupManager{}, tc.skipNodeMetrics)
			m.activePods = tc.activePods
			m.podsUnderRoot = tc.podsUnderRoot
			m.getNodeAllocatable = func() v1.ResourceList {
				return v1.ResourceList{v1.ResourceMemory: *resource.NewQuantity(tc.allocatable, resource.BinarySI)}
			}
			m.qosContainersInfo = QOSContainersInfo{
				Guaranteed: tc.root,
				Burstable:  NewCgroupName(tc.root, "burstable"),
				BestEffort: NewCgroupName(tc.root, "besteffort"),
			}
			configs := map[v1.PodQOSClass]*CgroupConfig{
				v1.PodQOSGuaranteed: {Name: m.qosContainersInfo.Guaranteed, ResourceParameters: &ResourceConfig{}},
				v1.PodQOSBurstable:  {Name: m.qosContainersInfo.Burstable, ResourceParameters: &ResourceConfig{}},
				v1.PodQOSBestEffort: {Name: m.qosContainersInfo.BestEffort, ResourceParameters: &ResourceConfig{}},
			}

			require.NoError(t, m.setCPUCgroupConfig(configs))
			m.setMemoryQoS(logger, configs)
			m.setMemoryReserve(logger, configs, percentReserve)

			if diff := cmp.Diff(tc.want, toQOSValues(t, configs)); diff != "" {
				t.Errorf("unexpected QoS cgroup values (-want +got):\n%s", diff)
			}

			var gotMetrics nodeMetrics
			var err error
			gotMetrics.MemoryMin, err = testutil.GetGaugeMetricValue(kubeletmetrics.MemoryQoSNodeMemoryMinBytes)
			require.NoError(t, err)
			gotMetrics.MemoryLow, err = testutil.GetGaugeMetricValue(kubeletmetrics.MemoryQoSNodeMemoryLowBytes)
			require.NoError(t, err)
			if diff := cmp.Diff(tc.wantMetrics, gotMetrics); diff != "" {
				t.Errorf("unexpected node-wide MemoryQoS metrics (-want +got):\n%s", diff)
			}
		})
	}
}

func TestQOSContainerManagerStartTakesPodsUnderRoot(t *testing.T) {
	inQoSCgroups := newPartitionTestPod(metav1.NamespaceDefault, "1/1Gi", "1/1Gi")
	inSibling := newPartitionTestPod("kube-system", "200m/128Mi", "200m/128Mi")
	activePods := func() []*v1.Pod { return []*v1.Pod{inQoSCgroups} }
	allPods := func() []*v1.Pod { return []*v1.Pod{inQoSCgroups, inSibling} }

	cases := []struct {
		name          string
		podsUnderRoot ActivePodsFunc
		want          []*v1.Pod
	}{
		{
			name:          "pods under the root are taken as given",
			podsUnderRoot: allPods,
			want:          []*v1.Pod{inQoSCgroups, inSibling},
		},
		{
			name: "without them, the pods under the root are the active pods",
			want: []*v1.Pod{inQoSCgroups},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			root := NewCgroupName(RootCgroupName, defaultNodeAllocatableCgroupName)
			m := newQOSContainerManager(nil, root, NodeConfig{CgroupsPerQOS: true}, &fakeCgroupManager{}, false)
			require.NoError(t, m.Start(ctx, func() v1.ResourceList { return v1.ResourceList{} }, activePods, tc.podsUnderRoot))

			assert.Equal(t, tc.want, m.getPodsUnderRoot())
			assert.Equal(t, []*v1.Pod{inQoSCgroups}, m.activePods())
		})
	}
}
