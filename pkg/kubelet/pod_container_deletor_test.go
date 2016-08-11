/*
Copyright 2016 The Kubernetes Authors.

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
	"reflect"
	"testing"
	"time"

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func testGetContainersToDeleteInPod(t *testing.T) {
	pod := kubecontainer.PodStatus{
		ContainerStatuses: []*kubecontainer.ContainerStatus{
			{
				ID:        kubecontainer.ContainerID{Type: "test", ID: "1"},
				Name:      "foo",
				CreatedAt: time.Now(),
				State:     kubecontainer.ContainerStateExited,
			},
			{
				ID:        kubecontainer.ContainerID{Type: "test", ID: "2"},
				Name:      "bar",
				CreatedAt: time.Now().Add(time.Second),
				State:     kubecontainer.ContainerStateExited,
			},
			{
				ID:        kubecontainer.ContainerID{Type: "test", ID: "3"},
				Name:      "bar",
				CreatedAt: time.Now().Add(2 * time.Second),
				State:     kubecontainer.ContainerStateExited,
			},
			{
				ID:        kubecontainer.ContainerID{Type: "test", ID: "4"},
				Name:      "bar",
				CreatedAt: time.Now().Add(3 * time.Second),
				State:     kubecontainer.ContainerStateExited,
			},
			{
				ID:        kubecontainer.ContainerID{Type: "test", ID: "5"},
				Name:      "bar",
				CreatedAt: time.Now().Add(4 * time.Second),
				State:     kubecontainer.ContainerStateRunning,
			},
		},
	}

	testCases := []struct {
		containersToKeep           int
		expectedContainersToDelete []*kubecontainer.ContainerStatus
	}{
		{
			0,
			[]*kubecontainer.ContainerStatus{pod.ContainerStatuses[3], pod.ContainerStatuses[2], pod.ContainerStatuses[1]},
		},
		{
			1,
			[]*kubecontainer.ContainerStatus{pod.ContainerStatuses[2], pod.ContainerStatuses[1]},
		},
		{
			2,
			[]*kubecontainer.ContainerStatus{pod.ContainerStatuses[1]},
		},
	}

	for _, test := range testCases {
		candidates := getContainersToDeleteInPod("4", &pod, test.containersToKeep)
		if !reflect.DeepEqual(getContainersToDeleteInPod("4", &pod, test.containersToKeep), test.expectedContainersToDelete) {
			t.Errorf("expected %v got %v", test.expectedContainersToDelete, candidates)
		}
	}
}
