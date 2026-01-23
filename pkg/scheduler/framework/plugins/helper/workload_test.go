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

package helper

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestMatchingWorkloadReference(t *testing.T) {
	workloadRef := func(name, podGroup, podGroupReplicaKey string) *v1.WorkloadReference {
		return &v1.WorkloadReference{
			Name:               name,
			PodGroup:           podGroup,
			PodGroupReplicaKey: podGroupReplicaKey,
		}
	}
	testCases := []struct {
		name     string
		pod1     *v1.Pod
		pod2     *v1.Pod
		expected bool
	}{
		{
			name:     "same pod with workloadRef",
			pod1:     st.MakePod().Name("pod1").Namespace("test").WorkloadRef(workloadRef("name", "pgName", "pgKey")).Obj(),
			pod2:     st.MakePod().Name("pod1").Namespace("test").WorkloadRef(workloadRef("name", "pgName", "pgKey")).Obj(),
			expected: true,
		},
		{
			name:     "different pods, same workloadRef",
			pod1:     st.MakePod().Name("pod1").Namespace("test").WorkloadRef(workloadRef("name", "pgName", "pgKey")).Obj(),
			pod2:     st.MakePod().Name("pod2").Namespace("test").WorkloadRef(workloadRef("name", "pgName", "pgKey")).Obj(),
			expected: true,
		},
		{
			name:     "same pod but no workloadRef",
			pod1:     st.MakePod().Name("pod1").Namespace("test").Obj(),
			pod2:     st.MakePod().Name("pod1").Namespace("test").Obj(),
			expected: false,
		},
		{
			name:     "different pods, only one with workloadRef",
			pod1:     st.MakePod().Name("pod1").Namespace("test").WorkloadRef(workloadRef("name", "pgName", "pgKey")).Obj(),
			pod2:     st.MakePod().Name("pod2").Namespace("test").Obj(),
			expected: false,
		},
		{
			name:     "same workloadRef but different namespaces",
			pod1:     st.MakePod().Name("pod1").Namespace("test1").WorkloadRef(workloadRef("name", "pgName", "pgKey")).Obj(),
			pod2:     st.MakePod().Name("pod2").Namespace("test2").WorkloadRef(workloadRef("name", "pgName", "pgKey")).Obj(),
			expected: false,
		},
		{
			name:     "same workload but different pod group",
			pod1:     st.MakePod().Name("pod1").Namespace("test").WorkloadRef(workloadRef("name", "pgName1", "pgKey")).Obj(),
			pod2:     st.MakePod().Name("pod2").Namespace("test").WorkloadRef(workloadRef("name", "pgName2", "pgKey")).Obj(),
			expected: false,
		},
		{
			name:     "same workload but different pod group replica key",
			pod1:     st.MakePod().Name("pod1").Namespace("test").WorkloadRef(workloadRef("name", "pgName", "pgKey1")).Obj(),
			pod2:     st.MakePod().Name("pod2").Namespace("test").WorkloadRef(workloadRef("name", "pgName", "pgKey2")).Obj(),
			expected: false,
		},
		{
			name:     "same pod group but different workload name",
			pod1:     st.MakePod().Name("pod1").Namespace("test").WorkloadRef(workloadRef("name1", "pgName", "pgKey")).Obj(),
			pod2:     st.MakePod().Name("pod2").Namespace("test").WorkloadRef(workloadRef("name2", "pgName", "pgKey")).Obj(),
			expected: false,
		},
	}
	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			if got := MatchingWorkloadReference(tt.pod1, tt.pod2); got != tt.expected {
				t.Errorf("MatchingWorkloadReference() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestPodGroupPolicy(t *testing.T) {
	workload := &schedulingapi.Workload{
		Spec: schedulingapi.WorkloadSpec{
			PodGroups: []schedulingapi.PodGroup{
				{
					Name: "pg1",
					Policy: schedulingapi.PodGroupPolicy{
						Gang: &schedulingapi.GangSchedulingPolicy{
							MinCount: 10,
						},
					},
				},
				{
					Name: "pg2",
					Policy: schedulingapi.PodGroupPolicy{
						Basic: &schedulingapi.BasicSchedulingPolicy{},
					},
				},
			},
		},
	}
	testCases := []struct {
		name           string
		podGroupName   string
		expectedPolicy schedulingapi.PodGroupPolicy
		expectedOk     bool
	}{
		{
			name:         "gang policy",
			podGroupName: "pg1",
			expectedPolicy: schedulingapi.PodGroupPolicy{
				Gang: &schedulingapi.GangSchedulingPolicy{
					MinCount: 10,
				},
			},
			expectedOk: true,
		},
		{
			name:         "basic policy",
			podGroupName: "pg2",
			expectedPolicy: schedulingapi.PodGroupPolicy{
				Basic: &schedulingapi.BasicSchedulingPolicy{},
			},
			expectedOk: true,
		},
		{
			name:           "pod group not found - return empty policy and false",
			podGroupName:   "pg3",
			expectedPolicy: schedulingapi.PodGroupPolicy{},
			expectedOk:     false,
		},
	}
	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := PodGroupPolicy(workload, tt.podGroupName)
			if ok != tt.expectedOk {
				t.Errorf("PodGroupPolicy() ok: %v, want: %v", ok, tt.expectedOk)
			}
			if diff := cmp.Diff(got, tt.expectedPolicy); diff != "" {
				t.Errorf("PodGroupPolicy() policy (-want,+got):\n%s", diff)
			}
		})
	}
}
