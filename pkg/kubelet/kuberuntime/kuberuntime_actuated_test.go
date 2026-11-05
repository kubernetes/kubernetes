/*
Copyright The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestActuatedStateMask(t *testing.T) {
	restartAlways := v1.ContainerRestartPolicyAlways
	resources := v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")}}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "uid", Name: "foo", Namespace: "ns"},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{{Name: "sidecar", Image: "busybox", RestartPolicy: &restartAlways, Resources: resources}},
			Containers: []v1.Container{{
				Name:         "main",
				Image:        "busybox",
				Resources:    resources,
				VolumeMounts: []v1.VolumeMount{{Name: "vol", MountPath: "/vol"}},
			}},
			EphemeralContainers: []v1.EphemeralContainer{{EphemeralContainerCommon: v1.EphemeralContainerCommon{
				Name:  "debug",
				Image: "busybox",
			}}},
			NodeName: "node",
			Volumes:  []v1.Volume{{Name: "vol", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}}},
		},
	}

	masked := pod.DeepCopy()
	DropVolumes(masked)
	// The volume manager records volumes as they are mounted; the rest of the spec is kept.
	want := pod.Spec.DeepCopy()
	want.Volumes = nil
	assert.Equal(t, *want, masked.Spec)
	assert.Equal(t, pod.ObjectMeta, masked.ObjectMeta)
}
