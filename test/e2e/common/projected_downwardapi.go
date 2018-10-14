/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("[sig-storage] Projected downwardAPI", func() {
	f := framework.NewDefaultFramework("projected")

	// How long to wait for a log pod to be displayed
	const podLogTimeout = 2 * time.Minute
	var podClient *framework.PodClient
	BeforeEach(func() {
		podClient = f.PodClient()
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, pod name
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests. Pod MUST be able to read the pod name from the mounted DownwardAPIVolumeFiles.
	*/
	framework.ConformanceIt("should provide podname only [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumePodForSimpleTest(podName, "/etc/podinfo/podname")

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("%s\n", podName),
		})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, volume mode 0400
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests. The default mode for the volume mount is set to 0400. Pod MUST be able to read the pod name from the mounted DownwardAPIVolumeFiles and the volume mode must be -r—-—————.
	*/
	framework.ConformanceIt("should set DefaultMode on files [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		defaultMode := int32(0400)
		pod := projectedDownwardAPIVolumePodForModeTest(podName, "/etc/podinfo/podname", nil, &defaultMode)

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			"mode of file \"/etc/podinfo/podname\": -r--------",
		})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, volume mode 0400
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests. The default mode for the volume mount is set to 0400. Pod MUST be able to read the pod name from the mounted DownwardAPIVolumeFiles and the volume mode must be -r—-—————.
	*/
	framework.ConformanceIt("should set mode on item file [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		mode := int32(0400)
		pod := projectedDownwardAPIVolumePodForModeTest(podName, "/etc/podinfo/podname", &mode, nil)

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			"mode of file \"/etc/podinfo/podname\": -r--------",
		})
	})

	It("should provide podname as non-root with fsgroup [NodeFeature:FSGroup]", func() {
		podName := "metadata-volume-" + string(uuid.NewUUID())
		uid := int64(1001)
		gid := int64(1234)
		pod := downwardAPIVolumePodForSimpleTest(podName, "/etc/podinfo/podname")
		pod.Spec.SecurityContext = &v1.PodSecurityContext{
			RunAsUser: &uid,
			FSGroup:   &gid,
		}
		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("%s\n", podName),
		})
	})

	It("should provide podname as non-root with fsgroup and defaultMode [NodeFeature:FSGroup]", func() {
		podName := "metadata-volume-" + string(uuid.NewUUID())
		uid := int64(1001)
		gid := int64(1234)
		mode := int32(0440) /* setting fsGroup sets mode to at least 440 */
		pod := projectedDownwardAPIVolumePodForModeTest(podName, "/etc/podinfo/podname", &mode, nil)
		pod.Spec.SecurityContext = &v1.PodSecurityContext{
			RunAsUser: &uid,
			FSGroup:   &gid,
		}
		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			"mode of file \"/etc/podinfo/podname\": -r--r-----",
		})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, update labels
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests and label items. Pod MUST be able to read the labels from the mounted DownwardAPIVolumeFiles. Labels are then updated. Pod MUST be able to read the updated values for the Labels.
	*/
	framework.ConformanceIt("should update labels on modification [NodeConformance]", func() {
		labels := map[string]string{}
		labels["key1"] = "value1"
		labels["key2"] = "value2"

		podName := "labelsupdate" + string(uuid.NewUUID())
		pod := projectedDownwardAPIVolumePodForUpdateTest(podName, labels, map[string]string{}, "/etc/podinfo/labels")
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

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, update annotation
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests and annotation items. Pod MUST be able to read the annotations from the mounted DownwardAPIVolumeFiles. Annotations are then updated. Pod MUST be able to read the updated values for the Annotations.
	*/
	framework.ConformanceIt("should update annotations on modification [NodeConformance]", func() {
		annotations := map[string]string{}
		annotations["builder"] = "bar"
		podName := "annotationupdate" + string(uuid.NewUUID())
		pod := projectedDownwardAPIVolumePodForUpdateTest(podName, map[string]string{}, annotations, "/etc/podinfo/annotations")

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

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, CPU limits
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests. Pod MUST be able to read the cpu limits from the mounted DownwardAPIVolumeFiles.
	*/
	framework.ConformanceIt("should provide container's cpu limit [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForContainerResources(podName, "/etc/podinfo/cpu_limit")

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("2\n"),
		})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, memory limits
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests. Pod MUST be able to read the memory limits from the mounted DownwardAPIVolumeFiles.
	*/
	framework.ConformanceIt("should provide container's memory limit [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForContainerResources(podName, "/etc/podinfo/memory_limit")

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("67108864\n"),
		})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, CPU request
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests. Pod MUST be able to read the cpu request from the mounted DownwardAPIVolumeFiles.
	*/
	framework.ConformanceIt("should provide container's cpu request [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForContainerResources(podName, "/etc/podinfo/cpu_request")

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("1\n"),
		})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, memory request
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests. Pod MUST be able to read the memory request from the mounted DownwardAPIVolumeFiles.
	*/
	framework.ConformanceIt("should provide container's memory request [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForContainerResources(podName, "/etc/podinfo/memory_request")

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("33554432\n"),
		})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, CPU limit, node allocatable
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests.  The CPU and memory resources for requests and limits are NOT specified for the container. Pod MUST be able to read the default cpu limits from the mounted DownwardAPIVolumeFiles.
	*/
	framework.ConformanceIt("should provide node allocatable (cpu) as default cpu limit if the limit is not set [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForDefaultContainerResources(podName, "/etc/podinfo/cpu_limit")

		f.TestContainerOutputRegexp("downward API volume plugin", pod, 0, []string{"[1-9]"})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, memory limit, node allocatable
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests.  The CPU and memory resources for requests and limits are NOT specified for the container. Pod MUST be able to read the default memory limits from the mounted DownwardAPIVolumeFiles.
	*/
	framework.ConformanceIt("should provide node allocatable (memory) as default memory limit if the limit is not set [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForDefaultContainerResources(podName, "/etc/podinfo/memory_limit")

		f.TestContainerOutputRegexp("downward API volume plugin", pod, 0, []string{"[1-9]"})
	})
})

func projectedDownwardAPIVolumePodForModeTest(name, filePath string, itemMode, defaultMode *int32) *v1.Pod {
	pod := projectedDownwardAPIVolumeBasePod(name, nil, nil)

	pod.Spec.Containers = []v1.Container{
		{
			Name:    "client-container",
			Image:   imageutils.GetE2EImage(imageutils.Mounttest),
			Command: []string{"/mounttest", "--file_mode=" + filePath},
			VolumeMounts: []v1.VolumeMount{
				{
					Name:      "podinfo",
					MountPath: "/etc/podinfo",
				},
			},
		},
	}
	if itemMode != nil {
		pod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].DownwardAPI.Items[0].Mode = itemMode
	}
	if defaultMode != nil {
		pod.Spec.Volumes[0].VolumeSource.Projected.DefaultMode = defaultMode
	}

	return pod
}

func projectedDownwardAPIVolumePodForUpdateTest(name string, labels, annotations map[string]string, filePath string) *v1.Pod {
	pod := projectedDownwardAPIVolumeBasePod(name, labels, annotations)

	pod.Spec.Containers = []v1.Container{
		{
			Name:    "client-container",
			Image:   imageutils.GetE2EImage(imageutils.Mounttest),
			Command: []string{"/mounttest", "--break_on_expected_content=false", "--retry_time=1200", "--file_content_in_loop=" + filePath},
			VolumeMounts: []v1.VolumeMount{
				{
					Name:      "podinfo",
					MountPath: "/etc/podinfo",
					ReadOnly:  false,
				},
			},
		},
	}

	applyLabelsAndAnnotationsToProjectedDownwardAPIPod(labels, annotations, pod)
	return pod
}

func projectedDownwardAPIVolumeBasePod(name string, labels, annotations map[string]string) *v1.Pod {
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
						Projected: &v1.ProjectedVolumeSource{
							Sources: []v1.VolumeProjection{
								{
									DownwardAPI: &v1.DownwardAPIProjection{
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
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	return pod
}

func applyLabelsAndAnnotationsToProjectedDownwardAPIPod(labels, annotations map[string]string, pod *v1.Pod) {
	if len(labels) > 0 {
		pod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].DownwardAPI.Items = append(pod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].DownwardAPI.Items, v1.DownwardAPIVolumeFile{
			Path: "labels",
			FieldRef: &v1.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "metadata.labels",
			},
		})
	}

	if len(annotations) > 0 {
		pod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].DownwardAPI.Items = append(pod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].DownwardAPI.Items, v1.DownwardAPIVolumeFile{
			Path: "annotations",
			FieldRef: &v1.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "metadata.annotations",
			},
		})
	}
}
