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
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
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

	expectedCandidates := []*kubecontainer.ContainerStatus{pod.ContainerStatuses[2], pod.ContainerStatuses[1]}
	candidates := newPodContainerDeletor(&containertest.FakeRuntime{}, 1).getContainersToDeleteInPod("2", &pod)
	if !reflect.DeepEqual(candidates, expectedCandidates) {
		t.Errorf("expected %v got %v", expectedCandidates, candidates)
	}
}
