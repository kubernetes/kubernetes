/*
Copyright 2018 The Kubernetes Authors.

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

package kubelet

import (
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
)

// fakeManager implements eviction Manager interface for testing purposes.
type fakeEvictionManager struct {
}

func NewFakeEvictionManager() eviction.Manager {
	return &fakeEvictionManager{}
}

func (s *fakeEvictionManager) Start(
	diskInfoProvider eviction.DiskInfoProvider,
	podFunc eviction.ActivePodsFunc,
	podCleanedUpFunc eviction.PodCleanedUpFunc,
	monitoringInterval time.Duration) {
}

func (s *fakeEvictionManager) IsUnderMemoryPressure() bool {
	return true
}

func (s *fakeEvictionManager) IsUnderDiskPressure() bool {
	return true
}

func (s *fakeEvictionManager) IsUnderPIDPressure() bool {
	return true
}

func TestSetNodeReadyCondition(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	kubelet.runtimeState.setRuntimeSync(time.Now())

	tests := []struct {
		name   string
		node   *v1.Node
		expect []v1.NodeCondition
	}{
		{
			name: "missing node capacity",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: testKubeletHostname},
				Spec:       v1.NodeSpec{},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:    v1.NodeOutOfDisk,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientDisk",
							Message: "kubelet has sufficient disk space available",
						},
						{
							Type:    v1.NodeMemoryPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientMemory",
							Message: "kubelet has sufficient memory available",
						},
					},
					Capacity: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(3000, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(20E9, resource.BinarySI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(2800, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(19900E6, resource.BinarySI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
				},
			},
			expect: []v1.NodeCondition{
				{
					Type:    v1.NodeOutOfDisk,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientDisk",
					Message: "kubelet has sufficient disk space available",
				},
				{
					Type:    v1.NodeMemoryPressure,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientMemory",
					Message: "kubelet has sufficient memory available",
				},
				{
					Type:    v1.NodeReady,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletNotReady",
					Message: "Missing node capacity for resources: pods",
				},
			},
		},
		{
			name: "append node ready condition",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: testKubeletHostname},
				Spec:       v1.NodeSpec{},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:    v1.NodeOutOfDisk,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientDisk",
							Message: "kubelet has sufficient disk space available",
						},
						{
							Type:    v1.NodeMemoryPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientMemory",
							Message: "kubelet has sufficient memory available",
						},
					},
					Capacity: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(3000, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(20E9, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(2800, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(19900E6, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
				},
			},
			expect: []v1.NodeCondition{
				{
					Type:    v1.NodeOutOfDisk,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientDisk",
					Message: "kubelet has sufficient disk space available",
				},
				{
					Type:    v1.NodeMemoryPressure,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientMemory",
					Message: "kubelet has sufficient memory available",
				},
				{
					Type:    v1.NodeReady,
					Status:  v1.ConditionTrue,
					Reason:  "KubeletReady",
					Message: "kubelet is posting ready status",
				},
			},
		},
		{
			name: "update node ready condition",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: testKubeletHostname},
				Spec:       v1.NodeSpec{},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:    v1.NodeOutOfDisk,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientDisk",
							Message: "kubelet has sufficient disk space available",
						},
						{
							Type:    v1.NodeMemoryPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientMemory",
							Message: "kubelet has sufficient memory available",
						},
						{
							Type:    v1.NodeReady,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletNotReady",
							Message: "Missing node capacity for resources: pods",
						},
					},
					Capacity: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(3000, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(20E9, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(2800, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(19900E6, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
				},
			},
			expect: []v1.NodeCondition{
				{
					Type:    v1.NodeOutOfDisk,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientDisk",
					Message: "kubelet has sufficient disk space available",
				},
				{
					Type:    v1.NodeMemoryPressure,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientMemory",
					Message: "kubelet has sufficient memory available",
				},
				{
					Type:    v1.NodeReady,
					Status:  v1.ConditionTrue,
					Reason:  "KubeletReady",
					Message: "kubelet is posting ready status",
				},
			},
		},
	}

	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			kubelet.setNodeReadyCondition(test.node)
			if !isConditionsEqual(test.node.Status.Conditions, test.expect) {
				t.Errorf("case[%d]:%s Expected Node Conditions: %v, Got Node Conditions: %v", i, test.name, test.expect, test.node.Status.Conditions)
			}
		})
	}
}

func TestSetNodeMemoryPressureCondition(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	fakeEvictionManager := NewFakeEvictionManager()
	kubelet.evictionManager = fakeEvictionManager

	tests := []struct {
		name   string
		node   *v1.Node
		expect []v1.NodeCondition
	}{
		{
			name: "append node MemoryPressure condition",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: testKubeletHostname},
				Spec:       v1.NodeSpec{},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:    v1.NodeOutOfDisk,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientDisk",
							Message: "kubelet has sufficient disk space available",
						},
						{
							Type:    v1.NodePIDPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientPID",
							Message: "kubelet has sufficient PID available",
						},
					},
					Capacity: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(3000, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(20E9, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(2800, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(19900E6, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
				},
			},
			expect: []v1.NodeCondition{
				{
					Type:    v1.NodeOutOfDisk,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientDisk",
					Message: "kubelet has sufficient disk space available",
				},
				{
					Type:    v1.NodePIDPressure,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientPID",
					Message: "kubelet has sufficient PID available",
				},
				{
					Type:    v1.NodeMemoryPressure,
					Status:  v1.ConditionTrue,
					Reason:  "KubeletHasInsufficientMemory",
					Message: "kubelet has insufficient memory available",
				},
			},
		},
		{
			name: "update node MemoryPressure condition",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: testKubeletHostname},
				Spec:       v1.NodeSpec{},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:    v1.NodeOutOfDisk,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientDisk",
							Message: "kubelet has sufficient disk space available",
						},
						{
							Type:    v1.NodePIDPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientPID",
							Message: "kubelet has sufficient PID available",
						},
						{
							Type:    v1.NodeMemoryPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientMemory",
							Message: "kubelet has sufficient memory available",
						},
					},
					Capacity: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(3000, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(20E9, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(2800, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(19900E6, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
				},
			},
			expect: []v1.NodeCondition{
				{
					Type:    v1.NodeOutOfDisk,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientDisk",
					Message: "kubelet has sufficient disk space available",
				},
				{
					Type:    v1.NodePIDPressure,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientPID",
					Message: "kubelet has sufficient PID available",
				},
				{
					Type:    v1.NodeMemoryPressure,
					Status:  v1.ConditionTrue,
					Reason:  "KubeletHasInsufficientMemory",
					Message: "kubelet has insufficient memory available",
				},
			},
		},
	}

	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			kubelet.setNodeMemoryPressureCondition(test.node)
			if !isConditionsEqual(test.node.Status.Conditions, test.expect) {
				t.Errorf("case[%d]:%s Expected Node Conditions: %v, Got Node Conditions: %v", i, test.name, test.expect, test.node.Status.Conditions)
			}
		})
	}
}

func TestSetNodePIDPressureCondition(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	fakeEvictionManager := NewFakeEvictionManager()
	kubelet.evictionManager = fakeEvictionManager

	tests := []struct {
		name   string
		node   *v1.Node
		expect []v1.NodeCondition
	}{
		{
			name: "append node PIDPressure condition",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: testKubeletHostname},
				Spec:       v1.NodeSpec{},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:    v1.NodeOutOfDisk,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientDisk",
							Message: "kubelet has sufficient disk space available",
						},
						{
							Type:    v1.NodeMemoryPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientMemory",
							Message: "kubelet has sufficient memory available",
						},
					},
					Capacity: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(3000, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(20E9, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(2800, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(19900E6, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
				},
			},
			expect: []v1.NodeCondition{
				{
					Type:    v1.NodeOutOfDisk,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientDisk",
					Message: "kubelet has sufficient disk space available",
				},
				{
					Type:    v1.NodeMemoryPressure,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientMemory",
					Message: "kubelet has sufficient memory available",
				},
				{
					Type:    v1.NodePIDPressure,
					Status:  v1.ConditionTrue,
					Reason:  "KubeletHasInsufficientPID",
					Message: "kubelet has insufficient PID available",
				},
			},
		},
		{
			name: "update node PIDPressure condition",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: testKubeletHostname},
				Spec:       v1.NodeSpec{},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:    v1.NodeOutOfDisk,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientDisk",
							Message: "kubelet has sufficient disk space available",
						},
						{
							Type:    v1.NodePIDPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientPID",
							Message: "kubelet has sufficient PID available",
						},
						{
							Type:    v1.NodeMemoryPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientMemory",
							Message: "kubelet has sufficient memory available",
						},
					},
					Capacity: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(3000, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(20E9, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(2800, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(19900E6, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
				},
			},
			expect: []v1.NodeCondition{
				{
					Type:    v1.NodeOutOfDisk,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientDisk",
					Message: "kubelet has sufficient disk space available",
				},
				{
					Type:    v1.NodePIDPressure,
					Status:  v1.ConditionTrue,
					Reason:  "KubeletHasInsufficientPID",
					Message: "kubelet has insufficient PID available",
				},
				{
					Type:    v1.NodeMemoryPressure,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientMemory",
					Message: "kubelet has sufficient memory available",
				},
			},
		},
	}

	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			kubelet.setNodePIDPressureCondition(test.node)
			if !isConditionsEqual(test.node.Status.Conditions, test.expect) {
				t.Errorf("case[%d]:%s Expected Node Conditions: %v, Got Node Conditions: %v", i, test.name, test.expect, test.node.Status.Conditions)
			}
		})
	}
}

func TestSetNodeDiskPressureCondition(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	fakeEvictionManager := NewFakeEvictionManager()
	kubelet.evictionManager = fakeEvictionManager

	tests := []struct {
		name   string
		node   *v1.Node
		expect []v1.NodeCondition
	}{
		{
			name: "append node DiskPressure condition",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: testKubeletHostname},
				Spec:       v1.NodeSpec{},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:    v1.NodePIDPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasInsufficientPID",
							Message: "kubelet has insufficient PID available",
						},
						{
							Type:    v1.NodeMemoryPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientMemory",
							Message: "kubelet has sufficient memory available",
						},
					},
					Capacity: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(3000, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(20E9, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(2800, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(19900E6, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
				},
			},
			expect: []v1.NodeCondition{
				{
					Type:    v1.NodePIDPressure,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasInsufficientPID",
					Message: "kubelet has insufficient PID available",
				},
				{
					Type:    v1.NodeMemoryPressure,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientMemory",
					Message: "kubelet has sufficient memory available",
				},
				{
					Type:    v1.NodeDiskPressure,
					Status:  v1.ConditionTrue,
					Reason:  "KubeletHasDiskPressure",
					Message: "kubelet has disk pressure",
				},
			},
		},
		{
			name: "update node DiskPressure condition",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: testKubeletHostname},
				Spec:       v1.NodeSpec{},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:    v1.NodePIDPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasInsufficientPID",
							Message: "kubelet has insufficient PID available",
						},
						{
							Type:    v1.NodeMemoryPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientMemory",
							Message: "kubelet has sufficient memory available",
						},
						{
							Type:    v1.NodeDiskPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasNoDiskPressure",
							Message: "kubelet has no disk pressure",
						},
					},
					Capacity: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(3000, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(20E9, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(2800, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(19900E6, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
				},
			},
			expect: []v1.NodeCondition{
				{
					Type:    v1.NodePIDPressure,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasInsufficientPID",
					Message: "kubelet has insufficient PID available",
				},
				{
					Type:    v1.NodeMemoryPressure,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientMemory",
					Message: "kubelet has sufficient memory available",
				},
				{
					Type:    v1.NodeDiskPressure,
					Status:  v1.ConditionTrue,
					Reason:  "KubeletHasDiskPressure",
					Message: "kubelet has disk pressure",
				},
			},
		},
	}

	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			kubelet.setNodeDiskPressureCondition(test.node)
			if !isConditionsEqual(test.node.Status.Conditions, test.expect) {
				t.Errorf("case[%d]:%s Expected Node Conditions: %v, Got Node Conditions: %v", i, test.name, test.expect, test.node.Status.Conditions)
			}
		})
	}
}

func TestSetNodeOODCondition(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	fakeEvictionManager := NewFakeEvictionManager()
	kubelet.evictionManager = fakeEvictionManager

	tests := []struct {
		name   string
		node   *v1.Node
		expect []v1.NodeCondition
	}{
		{
			name: "append node OutOfDisk condition",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: testKubeletHostname},
				Spec:       v1.NodeSpec{},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:    v1.NodePIDPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasInsufficientPID",
							Message: "kubelet has insufficient PID available",
						},
						{
							Type:    v1.NodeMemoryPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientMemory",
							Message: "kubelet has sufficient memory available",
						},
					},
					Capacity: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(3000, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(20E9, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(2800, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(19900E6, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
				},
			},
			expect: []v1.NodeCondition{
				{
					Type:    v1.NodePIDPressure,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasInsufficientPID",
					Message: "kubelet has insufficient PID available",
				},
				{
					Type:    v1.NodeMemoryPressure,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientMemory",
					Message: "kubelet has sufficient memory available",
				},
				{
					Type:    v1.NodeOutOfDisk,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientDisk",
					Message: "kubelet has sufficient disk space available",
				},
			},
		},
		{
			name: "update node OutOfDisk condition",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: testKubeletHostname},
				Spec:       v1.NodeSpec{},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:    v1.NodeOutOfDisk,
							Status:  v1.ConditionTrue,
							Reason:  "KubeletHasInSufficientDisk",
							Message: "kubelet has insufficient disk space available",
						},
						{
							Type:    v1.NodePIDPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasInsufficientPID",
							Message: "kubelet has insufficient PID available",
						},
						{
							Type:    v1.NodeMemoryPressure,
							Status:  v1.ConditionFalse,
							Reason:  "KubeletHasSufficientMemory",
							Message: "kubelet has sufficient memory available",
						},
					},
					Capacity: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(3000, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(20E9, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:              *resource.NewMilliQuantity(2800, resource.DecimalSI),
						v1.ResourceMemory:           *resource.NewQuantity(19900E6, resource.BinarySI),
						v1.ResourcePods:             *resource.NewQuantity(0, resource.DecimalSI),
						v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
					},
				},
			},
			expect: []v1.NodeCondition{
				{
					Type:    v1.NodeOutOfDisk,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientDisk",
					Message: "kubelet has sufficient disk space available",
				},
				{
					Type:    v1.NodePIDPressure,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasInsufficientPID",
					Message: "kubelet has insufficient PID available",
				},
				{
					Type:    v1.NodeMemoryPressure,
					Status:  v1.ConditionFalse,
					Reason:  "KubeletHasSufficientMemory",
					Message: "kubelet has sufficient memory available",
				},
			},
		},
	}

	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			kubelet.setNodeOODCondition(test.node)
			if !isConditionsEqual(test.node.Status.Conditions, test.expect) {
				t.Errorf("case[%d]:%s Expected Node Conditions: %v, Got Node Conditions: %v", i, test.name, test.expect, test.node.Status.Conditions)
			}
		})
	}
}

func isConditionsEqual(leftConditions, rightConditions []v1.NodeCondition) bool {
	for i := range leftConditions {
		if leftConditions[i].Type != rightConditions[i].Type || leftConditions[i].Status != rightConditions[i].Status ||
			leftConditions[i].Reason != rightConditions[i].Reason || leftConditions[i].Message != rightConditions[i].Message {
			return false
		}
	}
	return true
}
