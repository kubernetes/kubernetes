/*
<<<<<<< HEAD
Copyright The Kubernetes Authors.
=======
Copyright 2025 The Kubernetes Authors.
>>>>>>> cb7005383f6 (sync with v1alpha changes)

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
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func Test_IsSchedulableAfterPodGroupAdded(t *testing.T) {
	tests := []struct {
		name         string
		pod          *v1.Pod
		newPodGroup  *schedulingapi.PodGroup
		expectedHint fwk.QueueingHint
	}{
		{
			name:         "add a pod group which matches the pod's pod group name",
			pod:          st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPodGroup:  st.MakePodGroup().Name("pg").MinCount(1).TemplateRef("t", "w").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "add a pod group which doesn't match the pod's scheduling group name",
			pod:          st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPodGroup:  st.MakePodGroup().Name("pg2").MinCount(1).TemplateRef("t", "w").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "add a pod group which doesn't match the pod's scheduling group name",
			pod:          st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPodGroup:  st.MakePodGroup().Name("pg2").MinCount(1).TemplateRef("t", "w").Obj(),
			expectedHint: fwk.QueueSkip,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)

			actualHint, err := IsSchedulableAfterPodGroupAdded(logger, tc.pod, nil, tc.newPodGroup)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.expectedHint, actualHint); diff != "" {
				t.Errorf("Expected QueuingHint doesn't match (-want,+got):\n%s", diff)
			}
		})
	}
}

func Test_IsSchedulableAfterPodAdded(t *testing.T) {
	tests := []struct {
		name         string
		pod          *v1.Pod
		newPod       *v1.Pod
		expectedHint fwk.QueueingHint
	}{
		{
			name:         "add a newPod which matches the pod's scheduling group",
			pod:          st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPod:       st.MakePod().PodGroupName("pg").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "add a newPod which doesn't match the pod's namespace",
			pod:          st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPod:       st.MakePod().Namespace("foo").PodGroupName("pg").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "add a newPod which doesn't match the pod's pod group name",
			pod:          st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPod:       st.MakePod().PodGroupName("pg2").Obj(),
			expectedHint: fwk.QueueSkip,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)

			actualHint, err := IsSchedulableAfterPodAdded(logger, tc.pod, nil, tc.newPod)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.expectedHint, actualHint); diff != "" {
				t.Errorf("Expected QueuingHint doesn't match (-want,+got):\n%s", diff)
			}
		})
	}
}
