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
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Downward API volume", func() {
	// How long to wait for a log pod to be displayed
	const podLogTimeout = 2 * time.Minute
	f := framework.NewDefaultFramework("downward-api")
	var podClient *framework.PodClient
	BeforeEach(func() {
		podClient = f.PodClient()
	})

	It("should provide podname only [Conformance] [Volume]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumePodForSimpleTest(podName, "/etc/podname")

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("%s\n", podName),
		})
	})

	It("should set DefaultMode on files [Conformance] [Volume]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		defaultMode := int32(0400)
		pod := downwardAPIVolumePodForModeTest(podName, "/etc/podname", nil, &defaultMode)

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			"mode of file \"/etc/podname\": -r--------",
		})
	})

	It("should set mode on item file [Conformance] [Volume]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		mode := int32(0400)
		pod := downwardAPIVolumePodForModeTest(podName, "/etc/podname", &mode, nil)

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			"mode of file \"/etc/podname\": -r--------",
		})
	})

	It("should provide podname as non-root with fsgroup [Feature:FSGroup] [Volume]", func() {
		podName := "metadata-volume-" + string(uuid.NewUUID())
		uid := int64(1001)
		gid := int64(1234)
		pod := downwardAPIVolumePodForSimpleTest(podName, "/etc/podname")
		pod.Spec.SecurityContext = &v1.PodSecurityContext{
			RunAsUser: &uid,
			FSGroup:   &gid,
		}
		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("%s\n", podName),
		})
	})

	It("should provide podname as non-root with fsgroup and defaultMode [Feature:FSGroup] [Volume]", func() {
		podName := "metadata-volume-" + string(uuid.NewUUID())
		uid := int64(1001)
		gid := int64(1234)
		mode := int32(0440) /* setting fsGroup sets mode to at least 440 */
		pod := downwardAPIVolumePodForModeTest(podName, "/etc/podname", &mode, nil)
		pod.Spec.SecurityContext = &v1.PodSecurityContext{
			RunAsUser: &uid,
			FSGroup:   &gid,
		}
		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			"mode of file \"/etc/podname\": -r--r-----",
		})
	})

	It("should update labels on modification [Conformance] [Volume]", func() {
		labels := map[string]string{}
		labels["key1"] = "value1"
		labels["key2"] = "value2"

		podName := "labelsupdate" + string(uuid.NewUUID())
		pod := downwardAPIVolumePodForUpdateTest(podName, labels, map[string]string{}, "/etc/labels")
		containerName := "client-container"
		By("Creating the pod")
		podClient.CreateSync(pod)

		Eventually(func() (string, error) {
			return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, podName, containerName)
		},
			podLogTimeout, framework.Poll).Should(ContainSubstring("key1=\"value1\"\n"))

		//modify labels
		podClient.Update(podName, func(pod *v1.Pod) {
			pod.Labels["key3"] = "value3"
		})

		Eventually(func() (string, error) {
			return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, containerName)
		},
			podLogTimeout, framework.Poll).Should(ContainSubstring("key3=\"value3\"\n"))
	})

	It("should update annotations on modification [Conformance] [Volume]", func() {
		annotations := map[string]string{}
		annotations["builder"] = "bar"
		podName := "annotationupdate" + string(uuid.NewUUID())
		pod := downwardAPIVolumePodForUpdateTest(podName, map[string]string{}, annotations, "/etc/annotations")

		containerName := "client-container"
		By("Creating the pod")
		podClient.CreateSync(pod)

		pod, err := podClient.Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "Failed to get pod %q", pod.Name)

		Eventually(func() (string, error) {
			return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, containerName)
		},
			podLogTimeout, framework.Poll).Should(ContainSubstring("builder=\"bar\"\n"))

		//modify annotations
		podClient.Update(podName, func(pod *v1.Pod) {
			pod.Annotations["builder"] = "foo"
		})

		Eventually(func() (string, error) {
			return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, containerName)
		},
			podLogTimeout, framework.Poll).Should(ContainSubstring("builder=\"foo\"\n"))
	})

	It("should provide container's cpu limit [Conformance] [Volume]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForContainerResources(podName, "/etc/cpu_limit")

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("2\n"),
		})
	})

	It("should provide container's memory limit [Conformance] [Volume]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForContainerResources(podName, "/etc/memory_limit")

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("67108864\n"),
		})
	})

	It("should provide container's cpu request [Conformance] [Volume]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForContainerResources(podName, "/etc/cpu_request")

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("1\n"),
		})
	})

	It("should provide container's memory request [Conformance] [Volume]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForContainerResources(podName, "/etc/memory_request")

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("33554432\n"),
		})
	})

	It("should provide node allocatable (cpu) as default cpu limit if the limit is not set [Conformance] [Volume]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForDefaultContainerResources(podName, "/etc/cpu_limit")

		f.TestContainerOutputRegexp("downward API volume plugin", pod, 0, []string{"[1-9]"})
	})

	It("should provide node allocatable (memory) as default memory limit if the limit is not set [Conformance] [Volume]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForDefaultContainerResources(podName, "/etc/memory_limit")

		f.TestContainerOutputRegexp("downward API volume plugin", pod, 0, []string{"[1-9]"})
	})

})

func downwardAPIVolumePodForModeTest(name, filePath string, itemMode, defaultMode *int32) *v1.Pod {
	pod := downwardAPIVolumeBasePod(name, nil, nil)

	pod.Spec.Containers = []v1.Container{
		{
			Name:    "client-container",
			Image:   "gcr.io/google_containers/mounttest:0.8",
			Command: []string{"/mt", "--file_mode=" + filePath},
			VolumeMounts: []v1.VolumeMount{
				{
					Name:      "podinfo",
					MountPath: "/etc",
				},
			},
		},
	}
	if itemMode != nil {
		pod.Spec.Volumes[0].VolumeSource.DownwardAPI.Items[0].Mode = itemMode
	}
	if defaultMode != nil {
		pod.Spec.Volumes[0].VolumeSource.DownwardAPI.DefaultMode = defaultMode
	}

	return pod
}

func downwardAPIVolumePodForSimpleTest(name string, filePath string) *v1.Pod {
	pod := downwardAPIVolumeBasePod(name, nil, nil)

	pod.Spec.Containers = []v1.Container{
		{
			Name:    "client-container",
			Image:   "gcr.io/google_containers/mounttest:0.8",
			Command: []string{"/mt", "--file_content=" + filePath},
			VolumeMounts: []v1.VolumeMount{
				{
					Name:      "podinfo",
					MountPath: "/etc",
					ReadOnly:  false,
				},
			},
		},
	}

	return pod
}

func downwardAPIVolumeForContainerResources(name string, filePath string) *v1.Pod {
	pod := downwardAPIVolumeBasePod(name, nil, nil)
	pod.Spec.Containers = downwardAPIVolumeBaseContainers("client-container", filePath)
	return pod
}

func downwardAPIVolumeForDefaultContainerResources(name string, filePath string) *v1.Pod {
	pod := downwardAPIVolumeBasePod(name, nil, nil)
	pod.Spec.Containers = downwardAPIVolumeDefaultBaseContainer("client-container", filePath)
	return pod
}

func downwardAPIVolumeBaseContainers(name, filePath string) []v1.Container {
	return []v1.Container{
		{
			Name:    name,
			Image:   "gcr.io/google_containers/mounttest:0.8",
			Command: []string{"/mt", "--file_content=" + filePath},
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
			VolumeMounts: []v1.VolumeMount{
				{
					Name:      "podinfo",
					MountPath: "/etc",
					ReadOnly:  false,
				},
			},
		},
	}

}

func downwardAPIVolumeDefaultBaseContainer(name, filePath string) []v1.Container {
	return []v1.Container{
		{
			Name:    name,
			Image:   "gcr.io/google_containers/mounttest:0.8",
			Command: []string{"/mt", "--file_content=" + filePath},
			VolumeMounts: []v1.VolumeMount{
				{
					Name:      "podinfo",
					MountPath: "/etc",
				},
			},
		},
	}

}

func downwardAPIVolumePodForUpdateTest(name string, labels, annotations map[string]string, filePath string) *v1.Pod {
	pod := downwardAPIVolumeBasePod(name, labels, annotations)

	pod.Spec.Containers = []v1.Container{
		{
			Name:    "client-container",
			Image:   "gcr.io/google_containers/mounttest:0.8",
			Command: []string{"/mt", "--break_on_expected_content=false", "--retry_time=120", "--file_content_in_loop=" + filePath},
			VolumeMounts: []v1.VolumeMount{
				{
					Name:      "podinfo",
					MountPath: "/etc",
					ReadOnly:  false,
				},
			},
		},
	}

	applyLabelsAndAnnotationsToDownwardAPIPod(labels, annotations, pod)
	return pod
}

func downwardAPIVolumeBasePod(name string, labels, annotations map[string]string) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "podinfo",
					VolumeSource: v1.VolumeSource{
						DownwardAPI: &v1.DownwardAPIVolumeSource{
							Items: []v1.DownwardAPIVolumeFile{
								{
									Path: "podname",
									FieldRef: &v1.ObjectFieldSelector{
										APIVersion: "v1",
										FieldPath:  "metadata.name",
									},
								},
								{
									Path: "cpu_limit",
									ResourceFieldRef: &v1.ResourceFieldSelector{
										ContainerName: "client-container",
										Resource:      "limits.cpu",
									},
								},
								{
									Path: "cpu_request",
									ResourceFieldRef: &v1.ResourceFieldSelector{
										ContainerName: "client-container",
										Resource:      "requests.cpu",
									},
								},
								{
									Path: "memory_limit",
									ResourceFieldRef: &v1.ResourceFieldSelector{
										ContainerName: "client-container",
										Resource:      "limits.memory",
									},
								},
								{
									Path: "memory_request",
									ResourceFieldRef: &v1.ResourceFieldSelector{
										ContainerName: "client-container",
										Resource:      "requests.memory",
									},
								},
							},
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	return pod
}

func applyLabelsAndAnnotationsToDownwardAPIPod(labels, annotations map[string]string, pod *v1.Pod) {
	if len(labels) > 0 {
		pod.Spec.Volumes[0].DownwardAPI.Items = append(pod.Spec.Volumes[0].DownwardAPI.Items, v1.DownwardAPIVolumeFile{
			Path: "labels",
			FieldRef: &v1.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "metadata.labels",
			},
		})
	}

	if len(annotations) > 0 {
		pod.Spec.Volumes[0].DownwardAPI.Items = append(pod.Spec.Volumes[0].DownwardAPI.Items, v1.DownwardAPIVolumeFile{
			Path: "annotations",
			FieldRef: &v1.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "metadata.annotations",
			},
		})
	}
}

// TODO: add test-webserver example as pointed out in https://github.com/kubernetes/kubernetes/pull/5093#discussion-diff-37606771
