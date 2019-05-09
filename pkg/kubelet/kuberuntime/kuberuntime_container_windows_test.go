/*
Copyright 2019 The Kubernetes Authors.

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
	"github.com/stretchr/testify/assert"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

func TestDetermineEffectiveSecurityContext(t *testing.T) {
	containerName := "container_name"
	container := &corev1.Container{Name: containerName}
	dummyCredSpec := "test cred spec contents"

	buildPod := func(annotations map[string]string) *corev1.Pod {
		return &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: annotations,
			},
		}
	}

	t.Run("when there's a specific GMSA for that container, and no pod-wide GMSA", func(t *testing.T) {
		containerConfig := &runtimeapi.ContainerConfig{}

		pod := buildPod(map[string]string{
			"container_name.container.alpha.windows.kubernetes.io/gmsa-credential-spec": dummyCredSpec,
		})

		determineEffectiveSecurityContext(containerConfig, container, pod)

		assert.Equal(t, dummyCredSpec, containerConfig.Annotations["container.alpha.windows.kubernetes.io/gmsa-credential-spec"])
	})
	t.Run("when there's a specific GMSA for that container, and a pod-wide GMSA", func(t *testing.T) {
		containerConfig := &runtimeapi.ContainerConfig{}

		pod := buildPod(map[string]string{
			"container_name.container.alpha.windows.kubernetes.io/gmsa-credential-spec": dummyCredSpec,
			"pod.alpha.windows.kubernetes.io/gmsa-credential-spec":                      "should be ignored",
		})

		determineEffectiveSecurityContext(containerConfig, container, pod)

		assert.Equal(t, dummyCredSpec, containerConfig.Annotations["container.alpha.windows.kubernetes.io/gmsa-credential-spec"])
	})
	t.Run("when there's no specific GMSA for that container, and a pod-wide GMSA", func(t *testing.T) {
		containerConfig := &runtimeapi.ContainerConfig{}

		pod := buildPod(map[string]string{
			"pod.alpha.windows.kubernetes.io/gmsa-credential-spec": dummyCredSpec,
		})

		determineEffectiveSecurityContext(containerConfig, container, pod)

		assert.Equal(t, dummyCredSpec, containerConfig.Annotations["container.alpha.windows.kubernetes.io/gmsa-credential-spec"])
	})
	t.Run("when there's no specific GMSA for that container, and no pod-wide GMSA", func(t *testing.T) {
		containerConfig := &runtimeapi.ContainerConfig{}

		determineEffectiveSecurityContext(containerConfig, container, &corev1.Pod{})

		assert.Nil(t, containerConfig.Annotations)
	})
}
