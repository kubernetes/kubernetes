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

package e2e

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
)

var _ = Describe("Variable Expansion", func() {
	framework := NewFramework("var-expansion")

	It("should allow composing env vars into new env vars", func() {
		podName := "var-expansion-" + string(util.NewUUID())
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": podName},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    "dapi-container",
						Image:   "gcr.io/google_containers/busybox",
						Command: []string{"sh", "-c", "env"},
						Env: []api.EnvVar{
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
				RestartPolicy: api.RestartPolicyNever,
			},
		}

		framework.TestContainerOutput("env composition", pod, 0, []string{
			"FOO=foo-value",
			"BAR=bar-value",
			"FOOBAR=foo-value;;bar-value",
		})
	})

	It("should allow substituting values in a container's command", func() {
		podName := "var-expansion-" + string(util.NewUUID())
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": podName},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    "dapi-container",
						Image:   "gcr.io/google_containers/busybox",
						Command: []string{"sh", "-c", "TEST_VAR=wrong echo \"$(TEST_VAR)\""},
						Env: []api.EnvVar{
							{
								Name:  "TEST_VAR",
								Value: "test-value",
							},
						},
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}

		framework.TestContainerOutput("substitution in container's command", pod, 0, []string{
			"test-value",
		})
	})

	It("should allow substituting values in a container's args", func() {
		podName := "var-expansion-" + string(util.NewUUID())
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": podName},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    "dapi-container",
						Image:   "gcr.io/google_containers/busybox",
						Command: []string{"sh", "-c"},
						Args:    []string{"TEST_VAR=wrong echo \"$(TEST_VAR)\""},
						Env: []api.EnvVar{
							{
								Name:  "TEST_VAR",
								Value: "test-value",
							},
						},
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}

		framework.TestContainerOutput("substitution in container's args", pod, 0, []string{
			"test-value",
		})
	})
})
