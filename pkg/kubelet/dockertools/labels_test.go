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
	"k8s.io/kubernetes/pkg/api/testapi"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func getTestCase() (*api.Pod, *api.Container, int, labelledContainerInfo) {
	restartCount := 5
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
		Name: "test_container",
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
	expected := labelledContainerInfo{
		PodName:                   pod.Name,
		PodNamespace:              pod.Namespace,
		PodUID:                    pod.UID,
		PodDeletionGracePeriod:    pod.DeletionGracePeriodSeconds,
		PodTerminationGracePeriod: pod.Spec.TerminationGracePeriodSeconds,
		Name:                   container.Name,
		Hash:                   strconv.FormatUint(kubecontainer.HashContainer(container), 16),
		RestartCount:           restartCount,
		TerminationMessagePath: container.TerminationMessagePath,
		PreStopHandler:         container.Lifecycle.PreStop,
	}
	return pod, container, restartCount, expected
}

// Test whether we can get right information from label
func TestLabels(t *testing.T) {
	pod, container, restartCount, expected := getTestCase()
	labels := newLabels(container, pod, restartCount, false)
	containerInfo := getContainerInfoFromLabel(labels)
	if !reflect.DeepEqual(containerInfo, expected) {
		t.Errorf("expected %v, got %v", expected, containerInfo)
	}
}

// Test whether we can get information from v1.1 labels
func TestLabelsWithV11Labels(t *testing.T) {
	pod, container, restartCount, expected := getTestCase()

	// Test when DeletionGracePeriodSeconds, TerminationGracePeriodSeconds and Lifecycle are nil,
	// the information got from label should also be nil
	container.Lifecycle = nil
	pod.DeletionGracePeriodSeconds = nil
	pod.Spec.TerminationGracePeriodSeconds = nil
	expected.PodDeletionGracePeriod = nil
	expected.PodTerminationGracePeriod = nil
	expected.PreStopHandler = nil
	// Because container is changed, the Hash should be updated
	expected.Hash = strconv.FormatUint(kubecontainer.HashContainer(container), 16)
	labels := newLabels(container, pod, restartCount, false)
	containerInfo := getContainerInfoFromLabel(labels)
	if !reflect.DeepEqual(containerInfo, expected) {
		t.Errorf("expected %v, got %v", expected, containerInfo)
	}

	// Test when DeletionGracePeriodSeconds, TerminationGracePeriodSeconds and Lifecycle are nil,
	// but the old label kubernetesPodLabels are set.
	pod, container, _, _ = getTestCase()
	data, err := runtime.Encode(testapi.Default.Codec(), pod)
	if err != nil {
		t.Fatalf("Failed to encode pod %q into string: %v", format.Pod(pod), err)
	}
	labels[podLabel] = string(data)
	// When the label version is still currentLabelVersion, the information got from label should still
	// be nil.
	containerInfo = getContainerInfoFromLabel(labels)
	if !reflect.DeepEqual(containerInfo, expected) {
		t.Errorf("expected %v, got %v", expected, containerInfo)
	}
	// When the label version is not set (old version label), the information got from label should be set
	delete(labels, versionLabel)
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

// Test whether we can get information from v1.2 labels
func TestLabelsWithV12Labels(t *testing.T) {
	pod, container, restartCount, expected := getTestCase()

	// Test when RestartCount, TerminationMessagePath, DeletionGracePeriodSeconds, TerminationGracePeriodSeconds
	// and Lifecycle are not set, but corresponding old labels are set.
	labels := newLabels(container, pod, restartCount, false)
	labels[oldPodDeletionGracePeriodLabel] = labels[podDeletionGracePeriodLabel]
	delete(labels, podDeletionGracePeriodLabel)
	labels[oldPodTerminationGracePeriodLabel] = labels[podTerminationGracePeriodLabel]
	delete(labels, podTerminationGracePeriodLabel)
	labels[oldContainerPreStopHandlerLabel] = labels[containerPreStopHandlerLabel]
	delete(labels, containerPreStopHandlerLabel)
	labels[oldContainerRestartCountLabel] = labels[containerRestartCountLabel]
	delete(labels, containerRestartCountLabel)
	labels[oldContainerTerminationMessagePathLabel] = labels[containerTerminationMessagePathLabel]
	delete(labels, containerTerminationMessagePathLabel)
	// When the label version is not set (old version label), the information got from label should be set.
	delete(labels, versionLabel)
	containerInfo := getContainerInfoFromLabel(labels)
	if !reflect.DeepEqual(containerInfo, expected) {
		t.Errorf("expected %v, got %v", expected, containerInfo)
	}
	// When the label version is currentLabelVersion, the information got from label should still be empty.
	labels[versionLabel] = currentLabelVersion
	expected.PodDeletionGracePeriod = nil
	expected.PodTerminationGracePeriod = nil
	expected.PreStopHandler = nil
	expected.RestartCount = 0
	expected.TerminationMessagePath = ""
	containerInfo = getContainerInfoFromLabel(labels)
	if !reflect.DeepEqual(containerInfo, expected) {
		t.Errorf("expected %v, got %v", expected, containerInfo)
	}
}
