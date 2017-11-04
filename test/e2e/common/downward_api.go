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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	utilversion "k8s.io/kubernetes/pkg/util/version"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var (
	hostIPVersion = utilversion.MustParseSemantic("v1.8.0")
	podUIDVersion = utilversion.MustParseSemantic("v1.8.0")
)

var _ = Describe("[sig-api-machinery] Downward API", func() {
	f := framework.NewDefaultFramework("downward-api")

	framework.ConformanceIt("should provide pod name and namespace as env vars ", func() {
		podName := "downward-api-" + string(uuid.NewUUID())
		env := []v1.EnvVar{
			{
				Name: "POD_NAME",
				ValueFrom: &v1.EnvVarSource{
					FieldRef: &v1.ObjectFieldSelector{
						APIVersion: "v1",
						FieldPath:  "metadata.name",
					},
				},
			},
			{
				Name: "POD_NAMESPACE",
				ValueFrom: &v1.EnvVarSource{
					FieldRef: &v1.ObjectFieldSelector{
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

	framework.ConformanceIt("should provide pod IP as an env var ", func() {
		podName := "downward-api-" + string(uuid.NewUUID())
		env := []v1.EnvVar{
			{
				Name: "POD_IP",
				ValueFrom: &v1.EnvVarSource{
					FieldRef: &v1.ObjectFieldSelector{
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

	framework.ConformanceIt("should provide host IP as an env var ", func() {
		framework.SkipUnlessServerVersionGTE(hostIPVersion, f.ClientSet.Discovery())
		podName := "downward-api-" + string(uuid.NewUUID())
		env := []v1.EnvVar{
			{
				Name: "HOST_IP",
				ValueFrom: &v1.EnvVarSource{
					FieldRef: &v1.ObjectFieldSelector{
						APIVersion: "v1",
						FieldPath:  "status.hostIP",
					},
				},
			},
		}

		expectations := []string{
			"HOST_IP=(?:\\d+)\\.(?:\\d+)\\.(?:\\d+)\\.(?:\\d+)",
		}

		testDownwardAPI(f, podName, env, expectations)
	})

	framework.ConformanceIt("should provide container's limits.cpu/memory and requests.cpu/memory as env vars ", func() {
		podName := "downward-api-" + string(uuid.NewUUID())
		env := []v1.EnvVar{
			{
				Name: "CPU_LIMIT",
				ValueFrom: &v1.EnvVarSource{
					ResourceFieldRef: &v1.ResourceFieldSelector{
						Resource: "limits.cpu",
					},
				},
			},
			{
				Name: "MEMORY_LIMIT",
				ValueFrom: &v1.EnvVarSource{
					ResourceFieldRef: &v1.ResourceFieldSelector{
						Resource: "limits.memory",
					},
				},
			},
			{
				Name: "CPU_REQUEST",
				ValueFrom: &v1.EnvVarSource{
					ResourceFieldRef: &v1.ResourceFieldSelector{
						Resource: "requests.cpu",
					},
				},
			},
			{
				Name: "MEMORY_REQUEST",
				ValueFrom: &v1.EnvVarSource{
					ResourceFieldRef: &v1.ResourceFieldSelector{
						Resource: "requests.memory",
					},
				},
			},
		}
		expectations := []string{
			"CPU_LIMIT=2",
			"MEMORY_LIMIT=67108864",
			"CPU_REQUEST=1",
			"MEMORY_REQUEST=33554432",
		}

		testDownwardAPI(f, podName, env, expectations)
	})

	framework.ConformanceIt("should provide default limits.cpu/memory from node allocatable ", func() {
		podName := "downward-api-" + string(uuid.NewUUID())
		env := []v1.EnvVar{
			{
				Name: "CPU_LIMIT",
				ValueFrom: &v1.EnvVarSource{
					ResourceFieldRef: &v1.ResourceFieldSelector{
						Resource: "limits.cpu",
					},
				},
			},
			{
				Name: "MEMORY_LIMIT",
				ValueFrom: &v1.EnvVarSource{
					ResourceFieldRef: &v1.ResourceFieldSelector{
						Resource: "limits.memory",
					},
				},
			},
		}
		expectations := []string{
			"CPU_LIMIT=[1-9]",
			"MEMORY_LIMIT=[1-9]",
		}
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
						Env:     env,
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		testDownwardAPIUsingPod(f, pod, env, expectations)
	})

	framework.ConformanceIt("should provide pod UID as env vars ", func() {
		framework.SkipUnlessServerVersionGTE(podUIDVersion, f.ClientSet.Discovery())
		podName := "downward-api-" + string(uuid.NewUUID())
		env := []v1.EnvVar{
			{
				Name: "POD_UID",
				ValueFrom: &v1.EnvVarSource{
					FieldRef: &v1.ObjectFieldSelector{
						APIVersion: "v1",
						FieldPath:  "metadata.uid",
					},
				},
			},
		}

		expectations := []string{
			"POD_UID=[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}",
		}

		testDownwardAPI(f, podName, env, expectations)
	})
})

func testDownwardAPI(f *framework.Framework, podName string, env []v1.EnvVar, expectations []string) {
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
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("250m"),
							v1.ResourceMemory: resource.MustParse("32Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("1250m"),
							v1.ResourceMemory: resource.MustParse("64Mi"),
						},
					},
					Env: env,
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	testDownwardAPIUsingPod(f, pod, env, expectations)
}

func testDownwardAPIUsingPod(f *framework.Framework, pod *v1.Pod, env []v1.EnvVar, expectations []string) {
	f.TestContainerOutputRegexp("downward api env vars", pod, 0, expectations)
}
