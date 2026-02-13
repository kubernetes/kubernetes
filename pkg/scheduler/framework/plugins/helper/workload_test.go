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

	v1 "k8s.io/api/core/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/ptr"
)

func TestMatchingSchedulingGroup(t *testing.T) {
	schedulingGroup := func(podGroupName string) *v1.PodSchedulingGroup {
		return &v1.PodSchedulingGroup{
			PodGroupName: ptr.To(podGroupName),
		}
	}
	testCases := []struct {
		name     string
		pod1     *v1.Pod
		pod2     *v1.Pod
		expected bool
	}{
		{
			name:     "same pod with schedulingGroup",
			pod1:     st.MakePod().Name("pod1").Namespace("test").SchedulingGroup(schedulingGroup("name")).Obj(),
			pod2:     st.MakePod().Name("pod1").Namespace("test").SchedulingGroup(schedulingGroup("name")).Obj(),
			expected: true,
		},
		{
			name:     "different pods, same schedulingGroup",
			pod1:     st.MakePod().Name("pod1").Namespace("test").SchedulingGroup(schedulingGroup("name")).Obj(),
			pod2:     st.MakePod().Name("pod2").Namespace("test").SchedulingGroup(schedulingGroup("name")).Obj(),
			expected: true,
		},
		{
			name:     "same pod but no schedulingGroup",
			pod1:     st.MakePod().Name("pod1").Namespace("test").Obj(),
			pod2:     st.MakePod().Name("pod1").Namespace("test").Obj(),
			expected: false,
		},
		{
			name:     "different pods, only one with schedulingGroup",
			pod1:     st.MakePod().Name("pod1").Namespace("test").SchedulingGroup(schedulingGroup("name")).Obj(),
			pod2:     st.MakePod().Name("pod2").Namespace("test").Obj(),
			expected: false,
		},
		{
			name:     "same schedulingGroup but different namespaces",
			pod1:     st.MakePod().Name("pod1").Namespace("test1").SchedulingGroup(schedulingGroup("name")).Obj(),
			pod2:     st.MakePod().Name("pod2").Namespace("test2").SchedulingGroup(schedulingGroup("name")).Obj(),
			expected: false,
		},
		{
			name:     "same namespace but different pod group",
			pod1:     st.MakePod().Name("pod1").Namespace("test").SchedulingGroup(schedulingGroup("name1")).Obj(),
			pod2:     st.MakePod().Name("pod2").Namespace("test").SchedulingGroup(schedulingGroup("name2")).Obj(),
			expected: false,
		},
	}
	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			if got := MatchingSchedulingGroup(tt.pod1, tt.pod2); got != tt.expected {
				t.Errorf("MatchingSchedulingGroup() = %v, want %v", got, tt.expected)
			}
		})
	}
}
