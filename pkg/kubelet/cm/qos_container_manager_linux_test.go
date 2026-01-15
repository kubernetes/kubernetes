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
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	resource "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	cres "k8s.io/component-helpers/resource"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
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
	
	created []*CgroupConfig
	updates []*CgroupConfig
}

// Update() is the observation point for this test.
// Capture the updated cgroup config so it can be validated.
func (f *fakeCgroupManager) Update(logger klog.Logger, config *CgroupConfig) error{
	copiedConfig := *config
	f.updates = append(f.updates, &copiedConfig)
	return nil
}

// Create() must succeed for Start() to construct QoS cgroups.
// We do not assert on Create() behavior in this test.
func (f *fakeCgroupManager) Create(l klog.Logger, config *CgroupConfig) error {
	copiedConfig := *config
	f.created = append(f.created, &copiedConfig)
	return nil
}

func (f *fakeCgroupManager)	Destroy(l klog.Logger,config *CgroupConfig) error {return nil}
func (f *fakeCgroupManager)	Validate(name CgroupName) error {return nil}
func (f *fakeCgroupManager)	Exists(name CgroupName) bool {return false}
func (f *fakeCgroupManager)	Name(name CgroupName) string {return name.ToCgroupfs()}
func (f *fakeCgroupManager)	CgroupName(name string) CgroupName {return ParseCgroupfsToCgroupName(name)}
func (f *fakeCgroupManager)	Pids(logger klog.Logger, name CgroupName) []int {return nil}
func (f *fakeCgroupManager)	ReduceCPULimits(logger klog.Logger, cgroupName CgroupName) error {return nil}
func (f *fakeCgroupManager)	MemoryUsage(name CgroupName) (int64, error) {return int64(0),nil}
func (f *fakeCgroupManager)	GetCgroupConfig(name CgroupName, resource v1.ResourceName) (*ResourceConfig, error) {return nil,nil}
func (f *fakeCgroupManager)	SetCgroupConfig(logger klog.Logger, name CgroupName, resourceConfig *ResourceConfig) error {return nil}
func (f *fakeCgroupManager)	Version() int {return 1}

func expectedCPUShares(activeTestPods ActivePodsFunc) uint64 {
	var pods []*v1.Pod = activeTestPods()

	burstablePodCPURequest := int64(0)
	for i := range pods {
		pod := pods[i]
		qosClass := v1qos.GetPodQOS(pod)
		if qosClass != v1.PodQOSBurstable {
			// we only care about the burstable qos tier
			continue
		}	
		req := cres.PodRequests(pod, cres.PodResourcesOptions{})
		if request, found := req[v1.ResourceCPU]; found {
			burstablePodCPURequest += request.MilliValue()
		}
	}
	// set burstable shares based on current observe state
	burstableCPUShares := MilliCPUToShares(burstablePodCPURequest)
	return burstableCPUShares
}

func guaranteedTestPods() []*v1.Pod{
	return []*v1.Pod{
		
		{
			ObjectMeta: metav1.ObjectMeta{
				UID: types.UID(uuid.NewUUID()),
				Name: "guaranteed-pod",
				Namespace: "test",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "foo",
						Image: "busybox",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:resource.MustParse("1"),
							},
							Limits: v1.ResourceList{
								v1.ResourceCPU:resource.MustParse("1"),
							},
						},
					},
				},
			},
		},

	}
}

func burstableTestPods() []*v1.Pod{
	return []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				UID: types.UID(uuid.NewUUID()),
				Name: "burstable-pod",
				Namespace: "test",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "foo",
						Image: "busybox",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:resource.MustParse("1"),
							},
							Limits: v1.ResourceList{
								v1.ResourceCPU:resource.MustParse("2"),
							},
						},
					},
				},
			},
		},
	}
}

func bestEffortTestPods() []*v1.Pod{
	return []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				UID: types.UID(uuid.NewUUID()),
				Name: "besteffort-pod",
				Namespace: "test",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "foo",
						Image: "busybox",
					},
				},
			},
		},
	}
} 

func guaranteedAndBurstableTestPods() []*v1.Pod {
	pods := []*v1.Pod{}
	pods = append(pods, guaranteedTestPods()...)
	pods = append(pods, burstableTestPods()...)
	return pods
}
func besteffortAndBurstableTestPods() []*v1.Pod {
	pods := []*v1.Pod{}
	pods = append(pods, bestEffortTestPods()...)
	pods = append(pods, burstableTestPods()...)
	return pods
}

// TestQOSCPUConfigUpdate verifies that UpdateCgroups() computes and
// updates the correct CPU shares for each QoS class based on the
// currently active pods.
func TestQOSCPUConfigUpdate(t *testing.T){

	tests := []struct{
		name string
		testPods ActivePodsFunc
		expectedCPUShares uint64
	}{
		{
			name: "guaranteed-pods-only",
			testPods: guaranteedTestPods,
			expectedCPUShares:expectedCPUShares(guaranteedTestPods),
		},
		{
			name: "burstable-pods-only",
			testPods: burstableTestPods,
			expectedCPUShares: expectedCPUShares(burstableTestPods),
		},
		{
			name: "besteffort-pods-only",
			testPods: bestEffortTestPods,
			expectedCPUShares: expectedCPUShares(bestEffortTestPods),
		},
		{
			name: "guaranteed-and-burstable-pods",
			testPods: guaranteedAndBurstableTestPods,
			expectedCPUShares: expectedCPUShares(guaranteedAndBurstableTestPods),
		},
		{
			name: "besteffort-and-burstable-pods",
			testPods: besteffortAndBurstableTestPods,
			expectedCPUShares: expectedCPUShares(besteffortAndBurstableTestPods),
		},
	}

	for _,testCase := range tests{

		t.Run(testCase.name, func(t *testing.T) {

	testContainerManager,err := createTestQOSContainerManager(logr.Logger{})
	if err!=nil{
		t.Fatalf("Unable to create Test Qos Container Manager: %s", err)
		return
	}
	
	var fakecgroupManager = &fakeCgroupManager{}
	testContainerManager.cgroupManager = fakecgroupManager

	err=testContainerManager.Start(context.Background(), func() v1.ResourceList {return v1.ResourceList{}}, testCase.testPods)
	if err!=nil{
		t.Fatalf("Start() failed: %s",err)
	}

	// UpdateCgroups() is expected to update all QoS cgroups on each call
	// based on the current active pod set.
	err = testContainerManager.UpdateCgroups(logr.Logger{})
	if err!=nil{
		t.Fatalf("Error in UpdateCgroups(): %s", err)
	}

	//These flags will be used to check whether UpdateCgroups()
	//is updating the CPU shares for all QoS classes (cgroups) 
	foundBurstable := false
	foundBestEffort := false
	foundGuaranteed := false

	for _,config := range fakecgroupManager.updates {

		if strings.HasSuffix(config.Name.ToCgroupfs(), "burstable"){
			foundBurstable = true
			if *config.ResourceParameters.CPUShares != testCase.expectedCPUShares{
				t.Fatalf("Expected CPU Shares for Burstable: %d. Got: %d",testCase.expectedCPUShares, *config.ResourceParameters.CPUShares)
			}
			continue
		}

		if strings.HasSuffix(config.Name.ToCgroupfs(), "besteffort"){
			foundBestEffort = true
			if *config.ResourceParameters.CPUShares != MinShares {
				t.Fatalf("Expected CPU Shares for BestEffort: %d Got: %d", MinShares, *config.ResourceParameters.CPUShares)
			}
			continue
		}

		if config.Name.ToCgroupfs() == testContainerManager.cgroupRoot.ToCgroupfs() {
			foundGuaranteed = true
			if  config.ResourceParameters != nil && config.ResourceParameters.CPUShares != nil{
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