/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"testing"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/testapi"
)

func newPod(podName string, nodeName string, label map[string]string) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: testapi.Extensions.GroupVersion().String()},
		ObjectMeta: metav1.ObjectMeta{
			Labels:    label,
			Namespace: metav1.NamespaceDefault,
		},
		Spec: v1.PodSpec{
			NodeName: nodeName,
			Containers: []v1.Container{
				{
					Image: "foo/bar",
				},
			},
		},
	}
	pod.Name = podName
	return pod
}

func TestIsPodUpdated(t *testing.T) {
	templateGeneration := int64(12345)
	hash := "55555"
	labels := map[string]string{extensions.DaemonSetTemplateGenerationKey: fmt.Sprint(templateGeneration), extensions.DefaultDaemonSetUniqueLabelKey: hash}
	labelsNoHash := map[string]string{extensions.DaemonSetTemplateGenerationKey: fmt.Sprint(templateGeneration)}
	tests := []struct {
		test               string
		templateGeneration int64
		pod                *v1.Pod
		hash               string
		isUpdated          bool
	}{
		{
			"templateGeneration and hash both match",
			templateGeneration,
			newPod("pod1", "node1", labels),
			hash,
			true,
		},
		{
			"templateGeneration matches, hash doesn't",
			templateGeneration,
			newPod("pod1", "node1", labels),
			hash + "123",
			true,
		},
		{
			"templateGeneration matches, no hash label, has hash",
			templateGeneration,
			newPod("pod1", "node1", labelsNoHash),
			hash,
			true,
		},
		{
			"templateGeneration matches, no hash label, no hash",
			templateGeneration,
			newPod("pod1", "node1", labelsNoHash),
			"",
			true,
		},
		{
			"templateGeneration matches, has hash label, no hash",
			templateGeneration,
			newPod("pod1", "node1", labels),
			"",
			true,
		},
		{
			"templateGeneration doesn't match, hash does",
			templateGeneration + 1,
			newPod("pod1", "node1", labels),
			hash,
			true,
		},
		{
			"templateGeneration and hash don't match",
			templateGeneration + 1,
			newPod("pod1", "node1", labels),
			hash + "123",
			false,
		},
		{
			"empty labels, no hash",
			templateGeneration,
			newPod("pod1", "node1", map[string]string{}),
			"",
			false,
		},
		{
			"empty labels",
			templateGeneration,
			newPod("pod1", "node1", map[string]string{}),
			hash,
			false,
		},
		{
			"no labels",
			templateGeneration,
			newPod("pod1", "node1", nil),
			hash,
			false,
		},
	}
	for _, test := range tests {
		updated := IsPodUpdated(test.templateGeneration, test.pod, test.hash)
		if updated != test.isUpdated {
			t.Errorf("%s: IsPodUpdated returned wrong value. Expected %t, got %t", test.test, test.isUpdated, updated)
		}
	}
}

func TestCreatePodTemplate(t *testing.T) {
	tests := []struct {
		templateGeneration int64
		hash               string
		expectUniqueLabel  bool
	}{
		{int64(1), "", false},
		{int64(2), "3242341807", true},
	}
	for _, test := range tests {
		podTemplateSpec := v1.PodTemplateSpec{}
		newPodTemplate := CreatePodTemplate(podTemplateSpec, test.templateGeneration, test.hash)
		val, exists := newPodTemplate.ObjectMeta.Labels[extensions.DaemonSetTemplateGenerationKey]
		if !exists || val != fmt.Sprint(test.templateGeneration) {
			t.Errorf("Expected podTemplateSpec to have generation label value: %d, got: %s", test.templateGeneration, val)
		}
		val, exists = newPodTemplate.ObjectMeta.Labels[extensions.DefaultDaemonSetUniqueLabelKey]
		if test.expectUniqueLabel && (!exists || val != test.hash) {
			t.Errorf("Expected podTemplateSpec to have hash label value: %s, got: %s", test.hash, val)
		}
		if !test.expectUniqueLabel && exists {
			t.Errorf("Expected podTemplateSpec to have no hash label, got: %s", val)
		}
	}
}
