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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func getTestCase() (*api.Pod, *api.Container, *kubecontainer.ContainerStatus, containerInfo) {
	deletionGracePeriod := int64(10)
	terminationGracePeriod := int64(10)
	lifecycle := &api.Lifecycle{
		// Left PostStart as nil
		PreStop: &api.Handler{
			Exec: &api.ExecAction{
				Command: []string{"action1", "action2"},
			},
			HTTPGet: &api.HTTPGetAction{
				Path:   "path",
				Host:   "host",
				Port:   intstr.FromInt(8080),
				Scheme: "scheme",
			},
			TCPSocket: &api.TCPSocketAction{
				Port: intstr.FromString("80"),
			},
		},
	}
	container := &api.Container{
		Name:  "test_container",
		Ports: []api.ContainerPort{{Name: "test_port", ContainerPort: 8080, Protocol: api.ProtocolTCP}},
		TerminationMessagePath: "/somepath",
		Lifecycle:              lifecycle,
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "test_pod",
			Namespace: "test_pod_namespace",
			UID:       "test_pod_uid",
			DeletionGracePeriodSeconds: &deletionGracePeriod,
		},
		Spec: api.PodSpec{
			Containers:                    []api.Container{*container},
			TerminationGracePeriodSeconds: &terminationGracePeriod,
		},
	}
	lastStatus := &kubecontainer.ContainerStatus{
		RestartCount: 1,
		CreatedAt:    time.Now(),
		StartedAt:    time.Now(),
		FinishedAt:   time.Now(),
	}
	expected := containerInfo{
		containerMeta: containerMeta{
			PodName:      pod.Name,
			PodNamespace: pod.Namespace,
			PodUID:       pod.UID,
			Name:         container.Name,
			Hash:         strconv.FormatUint(kubecontainer.HashContainer(container), 16),
		},
		containerSpec: containerSpec{
			Ports: container.Ports,
			PodDeletionGracePeriod:    pod.DeletionGracePeriodSeconds,
			PodTerminationGracePeriod: pod.Spec.TerminationGracePeriodSeconds,
			TerminationMessagePath:    container.TerminationMessagePath,
			PreStopHandler:            container.Lifecycle.PreStop,
		},
		LastStatus: lastStatus,
	}
	return pod, container, lastStatus, expected
}

// Test whether we can get right information from label
func TestLabels(t *testing.T) {
	pod, container, lastStatus, expected := getTestCase()
	labels := newLabels(container, pod, lastStatus, false)
	info := getContainerInfoFromLabel(labels)
	if !reflect.DeepEqual(info, expected) {
		t.Errorf("expected %+v, got %+v", expected, info)
	}
}

// Test whether we can get information from v1.1 labels
func TestLabelsWithOldLabels(t *testing.T) {
	pod, container, lastStatus, expected := getTestCase()
	data, err := runtime.Encode(testapi.Default.Codec(), pod)
	if err != nil {
		t.Fatalf("Failed to encode pod %q into string: %v", format.Pod(pod), err)
	}

	// When no version label is set, only container meta should be got
	labels := newLabels(container, pod, lastStatus, false)
	delete(labels, versionLabel)
	info := getContainerInfoFromLabel(labels)
	e := containerInfo{containerMeta: expected.containerMeta}
	if !reflect.DeepEqual(info, e) {
		t.Errorf("expected %v, got %v", e, info)
	}

	// When no version is set, but the old labels are set, some information should be retrieved.
	labels[podLabel] = string(data)
	expected = containerInfo{
		containerMeta: expected.containerMeta,
		containerSpec: containerSpec{
			Ports: container.Ports,
			TerminationMessagePath:    container.TerminationMessagePath,
			PreStopHandler:            container.Lifecycle.PreStop,
			PodDeletionGracePeriod:    pod.DeletionGracePeriodSeconds,
			PodTerminationGracePeriod: pod.Spec.TerminationGracePeriodSeconds,
		},
	}
	info = getContainerInfoFromLabel(labels)
	if !reflect.DeepEqual(info, expected) {
		t.Errorf("expected %+v, got %+v", expected, info)
	}
}
