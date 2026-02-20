/*
Copyright 2024 The Kubernetes Authors.

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

package gpushare

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

func TestGPUShareFilter(t *testing.T) {
	// Setup a pod that requests 500 fractional GPU units
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "gpu-pod-1"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							GPUShareResourceName: resource.MustParse("500"),
						},
					},
				},
			},
		},
	}

	tests := []struct {
		name         string
		node         *v1.Node
		existingPods []*v1.Pod
		pod          *v1.Pod
		expectedCode fwk.Code
	}{
		{
			name: "node fits completely empty",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
				Status: v1.NodeStatus{
					Allocatable: v1.ResourceList{
						GPUShareResourceName: resource.MustParse("1000"),
					},
				},
			},
			existingPods: nil,
			pod:          pod,
			expectedCode: fwk.Success,
		},
		{
			name: "node already full",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node2"},
				Status: v1.NodeStatus{
					Allocatable: v1.ResourceList{
						GPUShareResourceName: resource.MustParse("1000"),
					},
				},
			},
			existingPods: []*v1.Pod{
				{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										GPUShareResourceName: resource.MustParse("600"),
									},
								},
							},
						},
					},
				},
			},
			pod:          pod,
			expectedCode: fwk.Unschedulable,
		},
		{
			name: "node has space",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node3"},
				Status: v1.NodeStatus{
					Allocatable: v1.ResourceList{
						GPUShareResourceName: resource.MustParse("1000"),
					},
				},
			},
			existingPods: []*v1.Pod{
				{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										GPUShareResourceName: resource.MustParse("400"),
									},
								},
							},
						},
					},
				},
			},
			pod:          pod,
			expectedCode: fwk.Success,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p := &GPUShare{}

			nodeInfo := framework.NewNodeInfo(test.existingPods...)
			nodeInfo.SetNode(test.node)

			status := p.Filter(context.Background(), nil, test.pod, nodeInfo)
			if status.Code() != test.expectedCode {
				t.Errorf("expected %v, got %v", test.expectedCode, status.Code())
			}
		})
	}
}

func TestGPUShareScore(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "gpu-pod-score"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							GPUShareResourceName: resource.MustParse("500"),
						},
					},
				},
			},
		},
	}

	tests := []struct {
		name          string
		node          *v1.Node
		existingPods  []*v1.Pod
		pod           *v1.Pod
		expectedScore int64
	}{
		{
			name: "node completely empty (score 50)",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
				Status: v1.NodeStatus{
					Allocatable: v1.ResourceList{
						GPUShareResourceName: resource.MustParse("1000"),
					},
				},
			},
			existingPods:  nil,
			pod:           pod,
			expectedScore: 50, // (500 + 0) / 1000 * 100
		},
		{
			name: "node partially full (score 90, bin-packing prefers this)",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node2"},
				Status: v1.NodeStatus{
					Allocatable: v1.ResourceList{
						GPUShareResourceName: resource.MustParse("1000"),
					},
				},
			},
			existingPods: []*v1.Pod{
				{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										GPUShareResourceName: resource.MustParse("400"),
									},
								},
							},
						},
					},
				},
			},
			pod:           pod,
			expectedScore: 90, // (500 + 400) / 1000 * 100
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p := &GPUShare{}

			nodeInfo := framework.NewNodeInfo(test.existingPods...)
			nodeInfo.SetNode(test.node)

			score, status := p.Score(context.Background(), nil, test.pod, nodeInfo)
			if status != nil && status.Code() != fwk.Success {
				t.Errorf("expected success, got %v", status)
			}
			if score != test.expectedScore {
				t.Errorf("expected score %d, got %d", test.expectedScore, score)
			}
		})
	}
}
