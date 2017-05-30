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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"

	"github.com/stretchr/testify/assert"
	"testing"
)

func TestVerifyRunAsNonRoot(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					Command:         []string{"testCommand"},
					WorkingDir:      "testWorkingDir",
				},
			},
		},
	}

	err := verifyRunAsNonRoot(pod, &pod.Spec.Containers[0], int64(0))
	assert.NoError(t, err)

	runAsUser := types.UnixUserID(0)
	RunAsNonRoot := false
	podWithContainerSecurityContext := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					Command:         []string{"testCommand"},
					WorkingDir:      "testWorkingDir",
					SecurityContext: &v1.SecurityContext{
						RunAsNonRoot: &RunAsNonRoot,
						RunAsUser:    &runAsUser,
					},
				},
			},
		},
	}

	err2 := verifyRunAsNonRoot(podWithContainerSecurityContext, &podWithContainerSecurityContext.Spec.Containers[0], int64(0))
	assert.EqualError(t, err2, "container's runAsUser breaks non-root policy")

	RunAsNonRoot = false
	podWithContainerSecurityContext = &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					Command:         []string{"testCommand"},
					WorkingDir:      "testWorkingDir",
					SecurityContext: &v1.SecurityContext{
						RunAsNonRoot: &RunAsNonRoot,
					},
				},
			},
		},
	}

	err3 := verifyRunAsNonRoot(podWithContainerSecurityContext, &podWithContainerSecurityContext.Spec.Containers[0], int64(0))
	assert.EqualError(t, err3, "container has runAsNonRoot and image will run as root")
}
