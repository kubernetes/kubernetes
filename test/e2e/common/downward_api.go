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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("Downward API", func() {
	f := framework.NewDefaultFramework("downward-api")

	It("should provide pod name and namespace as env vars [Conformance]", func() {
		podName := "downward-api-" + string(uuid.NewUUID())
		env := []api.EnvVar{
			{
				Name: "POD_NAME",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						APIVersion: "v1",
						FieldPath:  "metadata.name",
					},
				},
			},
			{
				Name: "POD_NAMESPACE",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						APIVersion: "v1",
						FieldPath:  "metadata.namespace",
					},
				},
			},
		}

		expectations := []string{
			fmt.Sprintf("POD_NAME=%v", podName),
			fmt.Sprintf("POD_NAMESPACE=%v", f.Namespace.Name),
		}

		testDownwardAPI(f, podName, env, expectations)
	})

	It("should provide pod IP as an env var [Conformance]", func() {
		podName := "downward-api-" + string(uuid.NewUUID())
		env := []api.EnvVar{
			{
				Name: "POD_IP",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						APIVersion: "v1",
						FieldPath:  "status.podIP",
					},
				},
			},
		}

		expectations := []string{
			"POD_IP=(?:\\d+)\\.(?:\\d+)\\.(?:\\d+)\\.(?:\\d+)",
		}

		testDownwardAPI(f, podName, env, expectations)
	})

	It("should provide container's limits.cpu/memory and requests.cpu/memory as env vars [Conformance]", func() {
		podName := "downward-api-" + string(uuid.NewUUID())
		env := []api.EnvVar{
			{
				Name: "CPU_LIMIT",
				ValueFrom: &api.EnvVarSource{
					ResourceFieldRef: &api.ResourceFieldSelector{
						Resource: "limits.cpu",
					},
				},
			},
			{
				Name: "MEMORY_LIMIT",
				ValueFrom: &api.EnvVarSource{
					ResourceFieldRef: &api.ResourceFieldSelector{
						Resource: "limits.memory",
					},
				},
			},
			{
				Name: "CPU_REQUEST",
				ValueFrom: &api.EnvVarSource{
					ResourceFieldRef: &api.ResourceFieldSelector{
						Resource: "requests.cpu",
					},
				},
			},
			{
				Name: "MEMORY_REQUEST",
				ValueFrom: &api.EnvVarSource{
					ResourceFieldRef: &api.ResourceFieldSelector{
						Resource: "requests.memory",
					},
				},
			},
		}
		expectations := []string{
			fmt.Sprintf("CPU_LIMIT=2"),
			fmt.Sprintf("MEMORY_LIMIT=67108864"),
			fmt.Sprintf("CPU_REQUEST=1"),
			fmt.Sprintf("MEMORY_REQUEST=33554432"),
		}

		testDownwardAPI(f, podName, env, expectations)
	})

	It("should provide default limits.cpu/memory from node allocatable [Conformance]", func() {
		podName := "downward-api-" + string(uuid.NewUUID())
		env := []api.EnvVar{
			{
				Name: "CPU_LIMIT",
				ValueFrom: &api.EnvVarSource{
					ResourceFieldRef: &api.ResourceFieldSelector{
						Resource: "limits.cpu",
					},
				},
			},
			{
				Name: "MEMORY_LIMIT",
				ValueFrom: &api.EnvVarSource{
					ResourceFieldRef: &api.ResourceFieldSelector{
						Resource: "limits.memory",
					},
				},
			},
		}
		expectations := []string{
			fmt.Sprintf("CPU_LIMIT=[1-9]"),
			fmt.Sprintf("MEMORY_LIMIT=[1-9]"),
		}
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": podName},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    "dapi-container",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"sh", "-c", "env"},
						Env:     env,
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}

		testDownwardAPIUsingPod(f, pod, env, expectations)
	})
})

func testDownwardAPI(f *framework.Framework, podName string, env []api.EnvVar, expectations []string) {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:   podName,
			Labels: map[string]string{"name": podName},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:    "dapi-container",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"sh", "-c", "env"},
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceCPU:    resource.MustParse("250m"),
							api.ResourceMemory: resource.MustParse("32Mi"),
						},
						Limits: api.ResourceList{
							api.ResourceCPU:    resource.MustParse("1250m"),
							api.ResourceMemory: resource.MustParse("64Mi"),
						},
					},
					Env: env,
				},
			},
			RestartPolicy: api.RestartPolicyNever,
		},
	}

	testDownwardAPIUsingPod(f, pod, env, expectations)
}

func testDownwardAPIUsingPod(f *framework.Framework, pod *api.Pod, env []api.EnvVar, expectations []string) {
	f.TestContainerOutputRegexp("downward api env vars", pod, 0, expectations)
}
