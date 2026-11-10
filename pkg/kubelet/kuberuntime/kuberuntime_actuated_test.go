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

// TestInitializeActuatedPodSeedsResources checks that the containers seeded from the allocated pod
// hold only their resources, and no image, so they aren't taken for a spec they were started with.
func TestInitializeActuatedPodSeedsResources(t *testing.T) {
	restartAlways := v1.ContainerRestartPolicyAlways
	resources := v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")}}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "uid", Name: "foo", Namespace: "ns"},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{{Name: "sidecar", Image: "busybox", RestartPolicy: &restartAlways, Resources: resources}},
			Containers:     []v1.Container{{Name: "main", Image: "busybox", Resources: resources}},
		},
	}
	setDynamicContainersGate(t, true)
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	m.InitializeActuatedPod(klog.Background(), pod)
	seeded, ok := m.actuatedState.GetContainer(pod.UID, "main", podutil.Containers)
	require.True(t, ok)
	assert.Equal(t, &v1.Container{Name: "main", Resources: resources}, seeded)
	seeded, ok = m.actuatedState.GetContainer(pod.UID, "sidecar", podutil.InitContainers)
	require.True(t, ok)
	assert.Equal(t, &v1.Container{Name: "sidecar", Resources: resources}, seeded)
}
