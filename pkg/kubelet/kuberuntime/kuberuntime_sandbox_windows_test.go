// +build windows

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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"testing"
)

func TestGeneratePodSandboxWindowsConfig(t *testing.T) {

	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)
	pod := newTestWindowsPod()

	wc, err := m.generatePodSandboxWindowsConfig(pod)
	assert.NoError(t, err)
	assert.Equal(t, wc.SecurityContext.RunAsUser, *pod.Spec.SecurityContext.WindowsOptions.RunAsUserName)
}

func newTestWindowsPod() *v1.Pod {
	toPtr := func(s string) *string {
		return &s
	}

	return &v1.Pod{
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
				},
			},
			SecurityContext: &v1.PodSecurityContext{
				WindowsOptions: &v1.WindowsSecurityContextOptions{
					RunAsUserName: toPtr("Container\tUser"),
				},
			},
		},
	}
}
