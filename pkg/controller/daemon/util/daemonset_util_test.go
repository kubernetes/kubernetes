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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions"
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
	tests := []struct {
		templateGeneration int64
		pod                *v1.Pod
		isUpdated          bool
	}{
		{
			int64(12345),
			newPod("pod1", "node1", map[string]string{extensions.DaemonSetTemplateGenerationKey: "12345"}),
			true,
		},
		{
			int64(12355),
			newPod("pod1", "node1", map[string]string{extensions.DaemonSetTemplateGenerationKey: "12345"}),
			false,
		},
		{
			int64(12355),
			newPod("pod1", "node1", map[string]string{}),
			false,
		},
		{
			int64(12355),
			newPod("pod1", "node1", nil),
			false,
		},
	}
	for _, test := range tests {
		updated := IsPodUpdated(test.templateGeneration, test.pod)
		if updated != test.isUpdated {
			t.Errorf("IsPodUpdated returned wrong value. Expected %t, got %t. TemplateGeneration: %d", test.isUpdated, updated, test.templateGeneration)
		}
	}
}

func TestGetPodTemplateWithGeneration(t *testing.T) {
	generation := int64(1)
	podTemplateSpec := v1.PodTemplateSpec{}
	newPodTemplate := GetPodTemplateWithGeneration(podTemplateSpec, generation)
	label, exists := newPodTemplate.ObjectMeta.Labels[extensions.DaemonSetTemplateGenerationKey]
	if !exists || label != fmt.Sprint(generation) {
		t.Errorf("Error in getting podTemplateSpec with label generation. Exists: %t, label: %s", exists, label)
	}
}
