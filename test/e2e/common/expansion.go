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

package common

import (
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

// These tests exercise the Kubernetes expansion syntax $(VAR).
// For more information, see: docs/design/expansion.md
var _ = framework.KubeDescribe("Variable Expansion", func() {
	f := framework.NewDefaultFramework("var-expansion")

	It("should allow composing env vars into new env vars [Conformance]", func() {
		podName := "var-expansion-" + string(uuid.NewUUID())
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": podName},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "dapi-container",
						Image:   busyboxImage,
						Command: []string{"sh", "-c", "env"},
						Env: []v1.EnvVar{
							{
								Name:  "FOO",
								Value: "foo-value",
							},
							{
								Name:  "BAR",
								Value: "bar-value",
							},
							{
								Name:  "FOOBAR",
								Value: "$(FOO);;$(BAR)",
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		f.TestContainerOutput("env composition", pod, 0, []string{
			"FOO=foo-value",
			"BAR=bar-value",
			"FOOBAR=foo-value;;bar-value",
		})
	})

	It("should allow substituting values in a container's command [Conformance]", func() {
		podName := "var-expansion-" + string(uuid.NewUUID())
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": podName},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "dapi-container",
						Image:   busyboxImage,
						Command: []string{"sh", "-c", "TEST_VAR=wrong echo \"$(TEST_VAR)\""},
						Env: []v1.EnvVar{
							{
								Name:  "TEST_VAR",
								Value: "test-value",
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		f.TestContainerOutput("substitution in container's command", pod, 0, []string{
			"test-value",
		})
	})

	It("should allow substituting values in a container's args [Conformance]", func() {
		podName := "var-expansion-" + string(uuid.NewUUID())
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": podName},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "dapi-container",
						Image:   busyboxImage,
						Command: []string{"sh", "-c"},
						Args:    []string{"TEST_VAR=wrong echo \"$(TEST_VAR)\""},
						Env: []v1.EnvVar{
							{
								Name:  "TEST_VAR",
								Value: "test-value",
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		f.TestContainerOutput("substitution in container's args", pod, 0, []string{
			"test-value",
		})
	})
})
