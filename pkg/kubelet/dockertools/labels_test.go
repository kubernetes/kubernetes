/*
Copyright 2015 The Kubernetes Authors.

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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

func TestLabels(t *testing.T) {
	restartCount := 5
	deletionGracePeriod := int64(10)
	terminationGracePeriod := int64(10)
	lifecycle := &v1.Lifecycle{
		// Left PostStart as nil
		PreStop: &v1.Handler{
			Exec: &v1.ExecAction{
				Command: []string{"action1", "action2"},
			},
			HTTPGet: &v1.HTTPGetAction{
				Path:   "path",
				Host:   "host",
				Port:   intstr.FromInt(8080),
				Scheme: "scheme",
			},
			TCPSocket: &v1.TCPSocketAction{
				Port: intstr.FromString("80"),
			},
		},
	}
	containerPorts := []v1.ContainerPort{
		{
			Name:          "http",
			HostPort:      80,
			ContainerPort: 8080,
			Protocol:      v1.ProtocolTCP,
		},
		{
			Name:          "https",
			HostPort:      443,
			ContainerPort: 6443,
			Protocol:      v1.ProtocolTCP,
		},
	}
	container := &v1.Container{
		Name:  "test_container",
		Ports: containerPorts,
		TerminationMessagePath: "/somepath",
		Lifecycle:              lifecycle,
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test_pod",
			Namespace: "test_pod_namespace",
			UID:       "test_pod_uid",
			DeletionGracePeriodSeconds: &deletionGracePeriod,
		},
		Spec: v1.PodSpec{
			Containers:                    []v1.Container{*container},
			TerminationGracePeriodSeconds: &terminationGracePeriod,
		},
	}
	expected := &labelledContainerInfo{
		PodName:                   pod.Name,
		PodNamespace:              pod.Namespace,
		PodUID:                    pod.UID,
		PodDeletionGracePeriod:    pod.DeletionGracePeriodSeconds,
		PodTerminationGracePeriod: pod.Spec.TerminationGracePeriodSeconds,
		Name:                   container.Name,
		Hash:                   strconv.FormatUint(kubecontainer.HashContainerLegacy(container), 16),
		RestartCount:           restartCount,
		TerminationMessagePath: container.TerminationMessagePath,
		PreStopHandler:         container.Lifecycle.PreStop,
		Ports:                  containerPorts,
	}

	// Test whether we can get right information from label
	labels := newLabels(container, pod, restartCount, false)
	containerInfo := getContainerInfoFromLabel(labels)
	if !reflect.DeepEqual(containerInfo, expected) {
		t.Errorf("expected %v, got %v", expected, containerInfo)
	}

	// Test when DeletionGracePeriodSeconds, TerminationGracePeriodSeconds and Lifecycle are nil,
	// the information got from label should also be nil
	container.Lifecycle = nil
	pod.DeletionGracePeriodSeconds = nil
	pod.Spec.TerminationGracePeriodSeconds = nil
	expected.PodDeletionGracePeriod = nil
	expected.PodTerminationGracePeriod = nil
	expected.PreStopHandler = nil
	// Because container is changed, the Hash should be updated
	expected.Hash = strconv.FormatUint(kubecontainer.HashContainerLegacy(container), 16)
	labels = newLabels(container, pod, restartCount, false)
	containerInfo = getContainerInfoFromLabel(labels)
	if !reflect.DeepEqual(containerInfo, expected) {
		t.Errorf("expected %v, got %v", expected, containerInfo)
	}

	// Test when DeletionGracePeriodSeconds, TerminationGracePeriodSeconds and Lifecycle are nil,
	// but the old label kubernetesPodLabel is set, the information got from label should also be set
	pod.DeletionGracePeriodSeconds = &deletionGracePeriod
	pod.Spec.TerminationGracePeriodSeconds = &terminationGracePeriod
	container.Lifecycle = lifecycle
	data, err := runtime.Encode(testapi.Default.Codec(), pod)
	if err != nil {
		t.Fatalf("Failed to encode pod %q into string: %v", format.Pod(pod), err)
	}
	labels[kubernetesPodLabel] = string(data)
	expected.PodDeletionGracePeriod = pod.DeletionGracePeriodSeconds
	expected.PodTerminationGracePeriod = pod.Spec.TerminationGracePeriodSeconds
	expected.PreStopHandler = container.Lifecycle.PreStop
	// Do not update expected.Hash here, because we directly use the labels in last test, so we never
	// changed the kubernetesContainerHashLabel in this test, the expected.Hash shouldn't be changed.
	containerInfo = getContainerInfoFromLabel(labels)
	if !reflect.DeepEqual(containerInfo, expected) {
		t.Errorf("expected %v, got %v", expected, containerInfo)
	}
}
