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

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	resource "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
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
		subsystems:    subsystems,
		cgroupManager: NewCgroupManager(logger, subsystems, "cgroupfs"),
		cgroupRoot:    cgroupRoot,
		qosReserved:   nil,
	}

	qosContainerManager.activePods = activeTestPods

	return qosContainerManager, nil
}

func TestQoSContainerCgroup(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	m, err := createTestQOSContainerManager(logger)
	assert.NoError(t, err)

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

	burstableMin := resource.MustParse("384Mi")
	guaranteedMin := resource.MustParse("128Mi")
	assert.Equal(t, qosConfigs[v1.PodQOSGuaranteed].ResourceParameters.Unified["memory.min"], strconv.FormatInt(burstableMin.Value()+guaranteedMin.Value(), 10))
	assert.Equal(t, qosConfigs[v1.PodQOSBurstable].ResourceParameters.Unified["memory.min"], strconv.FormatInt(burstableMin.Value(), 10))
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
func (f *fakeCgroupManager) Update(logger klog.Logger, config *CgroupConfig) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	copiedConfig := *config
	f.updates = append(f.updates, &copiedConfig)
	return nil
}

// Create() must succeed for Start() to construct QoS cgroups.
// We do not assert on Create() behavior in this test.
func (f *fakeCgroupManager) Create(l klog.Logger, config *CgroupConfig) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	copiedConfig := *config
	f.created = append(f.created, &copiedConfig)
	return nil
}

func (f *fakeCgroupManager) Destroy(l klog.Logger, config *CgroupConfig) error { return nil }
func (f *fakeCgroupManager) Validate(name CgroupName) error                    { return nil }
func (f *fakeCgroupManager) Exists(name CgroupName) bool                       { return false }
func (f *fakeCgroupManager) Name(name CgroupName) string                       { return name.ToCgroupfs() }
func (f *fakeCgroupManager) CgroupName(name string) CgroupName {
	return ParseCgroupfsToCgroupName(name)
}
func (f *fakeCgroupManager) Pids(logger klog.Logger, name CgroupName) []int { return nil }
func (f *fakeCgroupManager) ReduceCPULimits(logger klog.Logger, cgroupName CgroupName) error {
	return nil
}
func (f *fakeCgroupManager) MemoryUsage(name CgroupName) (int64, error) { return int64(0), nil }
func (f *fakeCgroupManager) GetCgroupConfig(name CgroupName, resource v1.ResourceName) (*ResourceConfig, error) {
	return nil, nil
}
func (f *fakeCgroupManager) SetCgroupConfig(logger klog.Logger, name CgroupName, resourceConfig *ResourceConfig) error {
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

			logger := logr.Discard()

			testContainerManager, err := createTestQOSContainerManager(logger)
			if err != nil {
				t.Fatalf("Unable to create Test Qos Container Manager: %s", err)
				return
			}

			fakecgroupManager := &fakeCgroupManager{}
			testContainerManager.cgroupManager = fakecgroupManager

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			err = testContainerManager.Start(ctx, func() v1.ResourceList { return v1.ResourceList{} }, testCase.testPods)

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

			// These flags will be used to check whether UpdateCgroups()
			// is updating the CPU shares for all QoS classes (cgroups)
			foundBurstable := false
			foundBestEffort := false
			foundGuaranteed := false

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

			if !foundBurstable {
				t.Fatalf("burstable cgroup not found")
			}
			if !foundBestEffort {
				t.Fatalf("besteffort cgroup not found")
			}
			if !foundGuaranteed {
				t.Fatalf("guaranteed cgroup not found")
			}

		})

	}
}
