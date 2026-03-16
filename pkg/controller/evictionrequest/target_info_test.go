/*
Copyright The Kubernetes Authors.

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

package evictionrequest

import (
	"testing"

	coordinationv1alpha1 "k8s.io/api/coordination/v1alpha1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
)

func makePodTarget(name, uid string) coordinationv1alpha1.EvictionTarget {
	return coordinationv1alpha1.EvictionTarget{
		Pod: &coordinationv1alpha1.PodReference{
			Name: name,
			UID:  types.UID(uid),
		},
	}
}

func testPod(name, uid string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			UID:  types.UID(uid),
		},
	}
}

func TestGetObjectMeta(t *testing.T) {
	tests := []struct {
		name     string
		target   targetInfo
		wantNil  bool
		wantName string
	}{
		{
			name:     "pod found",
			target:   newTargetInfo(makePodTarget("my-pod", "uid-1"), testPod("my-pod", "uid-1")),
			wantName: "my-pod",
		},
		{
			name:    "pod not found",
			target:  newTargetInfo(makePodTarget("my-pod", "uid-1"), nil),
			wantNil: true,
		},
		{
			name:    "empty target",
			target:  newTargetInfo(coordinationv1alpha1.EvictionTarget{}, nil),
			wantNil: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			meta := tt.target.GetObjectMeta()
			if tt.wantNil {
				if meta != nil {
					t.Errorf("expected nil, got %v", meta)
				}
				return
			}
			if meta == nil {
				t.Fatal("expected non-nil meta")
			}
			if got := meta.GetName(); got != tt.wantName {
				t.Errorf("GetName() = %q, want %q", got, tt.wantName)
			}
		})
	}
}

func TestIsGone(t *testing.T) {
	tests := []struct {
		name   string
		target targetInfo
		want   bool
	}{
		{
			name:   "pod found with matching UID",
			target: newTargetInfo(makePodTarget("my-pod", "uid-1"), testPod("my-pod", "uid-1")),
			want:   false,
		},
		{
			name:   "pod not found",
			target: newTargetInfo(makePodTarget("my-pod", "uid-1"), nil),
			want:   true,
		},
		{
			name:   "pod found with different UID",
			target: newTargetInfo(makePodTarget("my-pod", "uid-1"), testPod("my-pod", "uid-2")),
			want:   true,
		},
		{
			name:   "empty target",
			target: newTargetInfo(coordinationv1alpha1.EvictionTarget{}, nil),
			want:   false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.target.isGone(); got != tt.want {
				t.Errorf("isGone() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsPartOfPodGroup(t *testing.T) {
	tests := []struct {
		name   string
		target targetInfo
		want   bool
	}{
		{
			name:   "pod without PodGroup",
			target: newTargetInfo(makePodTarget("my-pod", "uid-1"), testPod("my-pod", "uid-1")),
			want:   false,
		},
		{
			name: "pod with PodGroup",
			target: func() targetInfo {
				pod := testPod("my-pod", "uid-1")
				pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: ptr.To("my-podgroup")}
				return newTargetInfo(makePodTarget("my-pod", "uid-1"), pod)
			}(),
			want: true,
		},
		{
			name:   "pod not found",
			target: newTargetInfo(makePodTarget("my-pod", "uid-1"), nil),
			want:   false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.target.hasSchedulingGroup(); got != tt.want {
				t.Errorf("hasSchedulingGroup() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsTerminal(t *testing.T) {
	tests := []struct {
		name   string
		target targetInfo
		want   bool
	}{
		{
			name:   "running pod",
			target: newTargetInfo(makePodTarget("my-pod", "uid-1"), testPod("my-pod", "uid-1")),
			want:   false,
		},
		{
			name: "succeeded pod",
			target: func() targetInfo {
				pod := testPod("my-pod", "uid-1")
				pod.Status.Phase = v1.PodSucceeded
				return newTargetInfo(makePodTarget("my-pod", "uid-1"), pod)
			}(),
			want: true,
		},
		{
			name: "failed pod",
			target: func() targetInfo {
				pod := testPod("my-pod", "uid-1")
				pod.Status.Phase = v1.PodFailed
				return newTargetInfo(makePodTarget("my-pod", "uid-1"), pod)
			}(),
			want: true,
		},
		{
			name:   "pod not found",
			target: newTargetInfo(makePodTarget("my-pod", "uid-1"), nil),
			want:   false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.target.isTerminal(); got != tt.want {
				t.Errorf("hasCompleted() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTargetType(t *testing.T) {
	tests := []struct {
		name   string
		target targetInfo
		want   string
	}{
		{
			name:   "pod target",
			target: newTargetInfo(makePodTarget("my-pod", "uid-1"), testPod("my-pod", "uid-1")),
			want:   "pod",
		},
		{
			name:   "empty target",
			target: newTargetInfo(coordinationv1alpha1.EvictionTarget{}, nil),
			want:   "unknown",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.target.targetType(); got.String() != tt.want {
				t.Errorf("targetType() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestEvictionInterceptors(t *testing.T) {
	tests := []struct {
		name   string
		target targetInfo
		want   []v1.EvictionInterceptor
	}{
		{
			name: "pod with interceptors",
			target: func() targetInfo {
				pod := testPod("my-pod", "uid-1")
				pod.Spec.EvictionInterceptors = []v1.EvictionInterceptor{
					{Name: "interceptor-a"},
					{Name: "interceptor-b"},
				}
				return newTargetInfo(makePodTarget("my-pod", "uid-1"), pod)
			}(),
			want: []v1.EvictionInterceptor{
				{Name: "interceptor-a"},
				{Name: "interceptor-b"},
			},
		},
		{
			name:   "pod without interceptors",
			target: newTargetInfo(makePodTarget("my-pod", "uid-1"), testPod("my-pod", "uid-1")),
			want:   nil,
		},
		{
			name:   "pod not found",
			target: newTargetInfo(makePodTarget("my-pod", "uid-1"), nil),
			want:   nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.target.evictionInterceptors()
			if len(got) != len(tt.want) {
				t.Fatalf("evictionInterceptors() returned %d items, want %d", len(got), len(tt.want))
			}
			for i := range got {
				if got[i].Name != tt.want[i].Name {
					t.Errorf("evictionInterceptors()[%d].Name = %q, want %q", i, got[i].Name, tt.want[i].Name)
				}
			}
		})
	}
}
