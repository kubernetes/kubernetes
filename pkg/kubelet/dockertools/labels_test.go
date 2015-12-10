/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package dockertools

import (
	"reflect"
	"strconv"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestLabels(t *testing.T) {
	restartCount := 5
	container := &api.Container{
		Name: "test_container",
		TerminationMessagePath: "/tmp",
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "test_pod",
			Namespace: "test_pod_namespace",
			UID:       "test_pod_uid",
		},
	}
	expected := &labelledContainerInfo{
		PodName:                pod.Name,
		PodNamespace:           pod.Namespace,
		PodUID:                 pod.UID,
		Name:                   container.Name,
		Hash:                   strconv.FormatUint(kubecontainer.HashContainer(container), 16),
		RestartCount:           restartCount,
		TerminationMessagePath: container.TerminationMessagePath,
	}

	labels := newLabels(container, pod, restartCount)
	containerInfo, err := getContainerInfoFromLabel(labels)
	if err != nil {
		t.Errorf("Unexpected error when getContainerInfoFromLabel: %v", err)
	}
	if !reflect.DeepEqual(containerInfo, expected) {
		t.Errorf("expected %v, got %v", expected, containerInfo)
	}
}
