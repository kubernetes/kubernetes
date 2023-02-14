/*
Copyright 2014 The Kubernetes Authors.

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

package debug

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func Test_formatTabularPods(t *testing.T) {
	tests := []struct {
		name string
		pods []v1.Pod
		want []string
	}{
		{
			want: []string{
				"POD PHASE START",
				"   NAME READY RESTARTS STATE TIME",
			},
		},
		{
			pods: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "blah"},
				},
			},
			want: []string{
				"POD  PHASE START",
				"   NAME READY RESTARTS STATE TIME",
				"test ",
			},
		},
		{
			pods: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "blah"},
					Status: v1.PodStatus{
						Phase: v1.PodFailed,
						InitContainerStatuses: []v1.ContainerStatus{
							{Name: "first", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{Reason: "Test"}}},
						},
						ContainerStatuses: []v1.ContainerStatus{
							{Name: "second", State: v1.ContainerState{Running: &v1.ContainerStateRunning{}}},
						},
						EphemeralContainerStatuses: []v1.ContainerStatus{
							{Name: "third", State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{Reason: "Test"}}},
						},
					},
				},
			},
			want: []string{
				"POD  PHASE START",
				"    NAME   READY RESTARTS STATE           TIME",
				"test Failed",
				"  I first  false 0        waiting:Test    ",
				"  C second false 0        running         0001-01-01T00:00:00Z",
				"  E third  false 0        terminated:Test 0001-01-01T00:00:00Z",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := formatTabularPods(tt.pods); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("unexpected lines: %s", cmp.Diff(tt.want, got))
			}
		})
	}
}
