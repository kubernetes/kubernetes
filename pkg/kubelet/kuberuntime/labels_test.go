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

package kuberuntime

import (
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestContainerLabels(t *testing.T) {
	deletionGracePeriod := int64(10)
	terminationGracePeriod := int64(10)
	lifecycle := &v1.Lifecycle{
		// Left PostStart as nil
		PreStop: &v1.LifecycleHandler{
			Exec: &v1.ExecAction{
				Command: []string{"action1", "action2"},
			},
			HTTPGet: &v1.HTTPGetAction{
				Path:   "path",
				Host:   "host",
				Port:   intstr.FromInt32(8080),
				Scheme: "scheme",
			},
			TCPSocket: &v1.TCPSocketAction{
				Port: intstr.FromString("80"),
			},
		},
	}
	container := &v1.Container{
		Name:                   "test_container",
		TerminationMessagePath: "/somepath",
		Lifecycle:              lifecycle,
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:                       "test_pod",
			Namespace:                  "test_pod_namespace",
			UID:                        "test_pod_uid",
			DeletionGracePeriodSeconds: &deletionGracePeriod,
		},
		Spec: v1.PodSpec{
			Containers:                    []v1.Container{*container},
			TerminationGracePeriodSeconds: &terminationGracePeriod,
		},
	}

	var tests = []struct {
		description string
		expected    *labeledContainerInfo
	}{
		{
			"Regular containers",
			&labeledContainerInfo{
				PodName:       pod.Name,
				PodNamespace:  pod.Namespace,
				PodUID:        pod.UID,
				ContainerName: container.Name,
			},
		},
	}

	// Test whether we can get right information from label
	for _, test := range tests {
		labels := newContainerLabels(container, pod)
		containerInfo := getContainerInfoFromLabels(labels)
		if !reflect.DeepEqual(containerInfo, test.expected) {
			t.Errorf("%v: expected %v, got %v", test.description, test.expected, containerInfo)
		}
	}
}

func TestContainerAnnotations(t *testing.T) {
	restartCount := 5
	deletionGracePeriod := int64(10)
	terminationGracePeriod := int64(10)
	opts := &kubecontainer.RunContainerOptions{
		Annotations: []kubecontainer.Annotation{
			{Name: "Foo", Value: "bar"},
		},
	}
	lifecycle := &v1.Lifecycle{
		// Left PostStart as nil
		PreStop: &v1.LifecycleHandler{
			Exec: &v1.ExecAction{
				Command: []string{"action1", "action2"},
			},
			HTTPGet: &v1.HTTPGetAction{
				Path:   "path",
				Host:   "host",
				Port:   intstr.FromInt32(8080),
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
		Name:                   "test_container",
		Ports:                  containerPorts,
		TerminationMessagePath: "/somepath",
		Lifecycle:              lifecycle,
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:                       "test_pod",
			Namespace:                  "test_pod_namespace",
			UID:                        "test_pod_uid",
			DeletionGracePeriodSeconds: &deletionGracePeriod,
		},
		Spec: v1.PodSpec{
			Containers:                    []v1.Container{*container},
			TerminationGracePeriodSeconds: &terminationGracePeriod,
		},
	}
	expected := &annotatedContainerInfo{
		ContainerPorts:            containerPorts,
		PodDeletionGracePeriod:    pod.DeletionGracePeriodSeconds,
		PodTerminationGracePeriod: pod.Spec.TerminationGracePeriodSeconds,
		Hash:                      kubecontainer.HashContainer(container),
		HashWithoutResources:      kubecontainer.HashContainerWithoutResources(container),
		RestartCount:              restartCount,
		TerminationMessagePath:    container.TerminationMessagePath,
		PreStopHandler:            container.Lifecycle.PreStop,
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)()
	// Test whether we can get right information from label
	annotations := newContainerAnnotations(container, pod, restartCount, opts)
	containerInfo := getContainerInfoFromAnnotations(annotations)
	if !reflect.DeepEqual(containerInfo, expected) {
		t.Errorf("expected %v, got %v", expected, containerInfo)
	}
	if v, ok := annotations[opts.Annotations[0].Name]; !ok || v != opts.Annotations[0].Value {
		t.Errorf("expected annotation %s to exist got %v, %v", opts.Annotations[0].Name, ok, v)
	}

	// Test when DeletionGracePeriodSeconds, TerminationGracePeriodSeconds and Lifecycle are nil,
	// the information got from annotations should also be nil
	container.Lifecycle = nil
	pod.DeletionGracePeriodSeconds = nil
	pod.Spec.TerminationGracePeriodSeconds = nil
	expected.PodDeletionGracePeriod = nil
	expected.PodTerminationGracePeriod = nil
	expected.PreStopHandler = nil
	// Because container is changed, the Hash should be updated
	expected.Hash = kubecontainer.HashContainer(container)
	expected.HashWithoutResources = kubecontainer.HashContainerWithoutResources(container)
	annotations = newContainerAnnotations(container, pod, restartCount, opts)
	containerInfo = getContainerInfoFromAnnotations(annotations)
	if !reflect.DeepEqual(containerInfo, expected) {
		t.Errorf("expected %v, got %v", expected, containerInfo)
	}
	if v, ok := annotations[opts.Annotations[0].Name]; !ok || v != opts.Annotations[0].Value {
		t.Errorf("expected annotation %s to exist got %v, %v", opts.Annotations[0].Name, ok, v)
	}
}

func TestPodLabels(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test_pod",
			Namespace: "test_pod_namespace",
			UID:       "test_pod_uid",
			Labels:    map[string]string{"foo": "bar"},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{},
		},
	}
	expected := &labeledPodSandboxInfo{
		Labels:       pod.Labels,
		PodName:      pod.Name,
		PodNamespace: pod.Namespace,
		PodUID:       pod.UID,
	}

	// Test whether we can get right information from label
	labels := newPodLabels(pod)
	podSandboxInfo := getPodSandboxInfoFromLabels(labels)
	if !reflect.DeepEqual(podSandboxInfo, expected) {
		t.Errorf("expected %v, got %v", expected, podSandboxInfo)
	}
}

func TestPodAnnotations(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test_pod",
			Namespace:   "test_pod_namespace",
			UID:         "test_pod_uid",
			Annotations: map[string]string{"foo": "bar"},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{},
		},
	}
	expected := &annotatedPodSandboxInfo{
		Annotations: map[string]string{"foo": "bar"},
	}

	// Test whether we can get right information from annotations
	annotations := newPodAnnotations(pod)
	podSandboxInfo := getPodSandboxInfoFromAnnotations(annotations)
	if !reflect.DeepEqual(podSandboxInfo, expected) {
		t.Errorf("expected %v, got %v", expected, podSandboxInfo)
	}
}
