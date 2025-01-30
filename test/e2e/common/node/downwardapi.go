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

package node

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
	e2epodoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("Downward API", func() {
	f := framework.NewDefaultFramework("downward-api")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	/*
	   Release: v1.9
	   Testname: DownwardAPI, environment for name, namespace and ip
	   Description: Downward API MUST expose Pod and Container fields as environment variables. Specify Pod Name, namespace and IP as environment variable in the Pod Spec are visible at runtime in the container.
	*/
	framework.ConformanceIt("should provide pod name, namespace and IP address as env vars", f.WithNodeConformance(), func(ctx context.Context) {
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
			fmt.Sprintf("POD_NAME=%v", podName),
			fmt.Sprintf("POD_NAMESPACE=%v", f.Namespace.Name),
			fmt.Sprintf("POD_IP=%v|%v", e2enetwork.RegexIPv4, e2enetwork.RegexIPv6),
		}

		testDownwardAPI(ctx, f, podName, env, expectations)
	})

	/*
	   Release: v1.9
	   Testname: DownwardAPI, environment for host ip
	   Description: Downward API MUST expose Pod and Container fields as environment variables. Specify host IP as environment variable in the Pod Spec are visible at runtime in the container.
	*/
	framework.ConformanceIt("should provide host IP as an env var", f.WithNodeConformance(), func(ctx context.Context) {
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
			fmt.Sprintf("HOST_IP=%v|%v", e2enetwork.RegexIPv4, e2enetwork.RegexIPv6),
		}

		testDownwardAPI(ctx, f, podName, env, expectations)
	})

	/*
	   Release: v1.32
	   Testname: DownwardAPI, environment for hostIPs
	   Description: Downward API MUST expose Pod and Container fields as environment variables. Specify hostIPs as environment variable in the Pod Spec are visible at runtime in the container.
	*/
	framework.ConformanceIt("should provide hostIPs as an env var", f.WithNodeConformance(), func(ctx context.Context) {
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
			fmt.Sprintf("HOST_IP=%v|%v", e2enetwork.RegexIPv4, e2enetwork.RegexIPv6),
		}

		testDownwardAPI(ctx, f, podName, env, expectations)
	})

	ginkgo.It("should provide host IP and pod IP as an env var if pod uses host network [LinuxOnly]", func(ctx context.Context) {
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
			fmt.Sprintf("OK"),
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
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", `[[ "${HOST_IP:?}" == "${POD_IP:?}" ]] && echo 'OK' || echo "HOST_IP: '${HOST_IP}' != POD_IP: '${POD_IP}'"`},
						Env:     env,
					},
				},
				HostNetwork:   true,
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		testDownwardAPIUsingPod(ctx, f, pod, env, expectations)

	})

	/*
	   Release: v1.9
	   Testname: DownwardAPI, environment for CPU and memory limits and requests
	   Description: Downward API MUST expose CPU request and Memory request set through environment variables at runtime in the container.
	*/
	framework.ConformanceIt("should provide container's limits.cpu/memory and requests.cpu/memory as env vars", f.WithNodeConformance(), func(ctx context.Context) {
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

		testDownwardAPI(ctx, f, podName, env, expectations)
	})

	/*
	   Release: v1.9
	   Testname: DownwardAPI, environment for default CPU and memory limits and requests
	   Description: Downward API MUST expose CPU request and Memory limits set through environment variables at runtime in the container.
	*/
	framework.ConformanceIt("should provide default limits.cpu/memory from node allocatable", f.WithNodeConformance(), func(ctx context.Context) {
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
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "env"},
						Env:     env,
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		testDownwardAPIUsingPod(ctx, f, pod, env, expectations)
	})

	/*
	   Release: v1.9
	   Testname: DownwardAPI, environment for Pod UID
	   Description: Downward API MUST expose Pod UID set through environment variables at runtime in the container.
	*/
	framework.ConformanceIt("should provide pod UID as env vars", f.WithNodeConformance(), func(ctx context.Context) {
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

		testDownwardAPI(ctx, f, podName, env, expectations)
	})
})

var _ = SIGDescribe("Downward API", framework.WithSerial(), framework.WithDisruptive(), feature.DownwardAPIHugePages, func() {
	f := framework.NewDefaultFramework("downward-api")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("Downward API tests for hugepages", func() {
		ginkgo.It("should provide container's limits.hugepages-<pagesize> and requests.hugepages-<pagesize> as env vars", func(ctx context.Context) {
			podName := "downward-api-" + string(uuid.NewUUID())
			env := []v1.EnvVar{
				{
					Name: "HUGEPAGES_LIMIT",
					ValueFrom: &v1.EnvVarSource{
						ResourceFieldRef: &v1.ResourceFieldSelector{
							Resource: "limits.hugepages-2Mi",
						},
					},
				},
				{
					Name: "HUGEPAGES_REQUEST",
					ValueFrom: &v1.EnvVarSource{
						ResourceFieldRef: &v1.ResourceFieldSelector{
							Resource: "requests.hugepages-2Mi",
						},
					},
				},
			}

			// Important: we explicitly request no hugepages so the test can run where none are present.
			expectations := []string{
				fmt.Sprintf("HUGEPAGES_LIMIT=%d", 0),
				fmt.Sprintf("HUGEPAGES_REQUEST=%d", 0),
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
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"sh", "-c", "env"},
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									"cpu":           resource.MustParse("10m"),
									"hugepages-2Mi": resource.MustParse("0Mi"),
								},
								Limits: v1.ResourceList{
									"hugepages-2Mi": resource.MustParse("0Mi"),
								},
							},
							Env: env,
						},
					},
					RestartPolicy: v1.RestartPolicyNever,
				},
			}
			testDownwardAPIUsingPod(ctx, f, pod, env, expectations)
		})

		ginkgo.It("should provide default limits.hugepages-<pagesize> from node allocatable", func(ctx context.Context) {
			podName := "downward-api-" + string(uuid.NewUUID())
			env := []v1.EnvVar{
				{
					Name: "HUGEPAGES_LIMIT",
					ValueFrom: &v1.EnvVarSource{
						ResourceFieldRef: &v1.ResourceFieldSelector{
							Resource: "limits.hugepages-2Mi",
						},
					},
				},
			}
			// Important: we allow for 0 so the test passes in environments where no hugepages are allocated.
			expectations := []string{
				"HUGEPAGES_LIMIT=((0)|([1-9][0-9]*))\n",
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
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"sh", "-c", "env"},
							Env:     env,
						},
					},
					RestartPolicy: v1.RestartPolicyNever,
				},
			}

			testDownwardAPIUsingPod(ctx, f, pod, env, expectations)
		})
	})

})

func testDownwardAPI(ctx context.Context, f *framework.Framework, podName string, env []v1.EnvVar, expectations []string) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   podName,
			Labels: map[string]string{"name": podName},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "dapi-container",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
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

	testDownwardAPIUsingPod(ctx, f, pod, env, expectations)
}

func testDownwardAPIUsingPod(ctx context.Context, f *framework.Framework, pod *v1.Pod, env []v1.EnvVar, expectations []string) {
	e2epodoutput.TestContainerOutputRegexp(ctx, f, "downward api env vars", pod, 0, expectations)
}
