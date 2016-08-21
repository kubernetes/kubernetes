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

package e2e_node

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Kubelet Cgroup Manager [Skip]", func() {
	f := framework.NewDefaultFramework("kubelet-cgroup-manager")
	Describe("QOS containers", func() {
		Context("On enabling QOS cgroup hierarchy", func() {
			It("Top level QoS containers should have been created", func() {
				if framework.TestContext.CgroupsPerQOS {
					podName := "qos-pod" + string(uuid.NewUUID())
					contName := "qos-container" + string(uuid.NewUUID())
					pod := &api.Pod{
						ObjectMeta: api.ObjectMeta{
							Name: podName,
						},
						Spec: api.PodSpec{
							// Don't restart the Pod since it is expected to exit
							RestartPolicy: api.RestartPolicyNever,
							Containers: []api.Container{
								{
									Image: ImageRegistry[busyBoxImage],
									Name:  contName,
									// TODO(@dubstack) Make this and the subsequent e2e tests work on systemd systems
									Command: []string{"sh", "-c", "if [ -d /tmp/memory/Burstable ] && [ -d /tmp/memory/BestEffort ]; then exit 0; else exit 1; fi"},
									VolumeMounts: []api.VolumeMount{
										{
											Name:      "sysfscgroup",
											MountPath: "/tmp",
										},
									},
								},
							},
							Volumes: []api.Volume{
								{
									Name: "sysfscgroup",
									VolumeSource: api.VolumeSource{
										HostPath: &api.HostPathVolumeSource{Path: "/sys/fs/cgroup"},
									},
								},
							},
						},
					}
					podClient := f.PodClient()
					podClient.Create(pod)
					err := framework.WaitForPodSuccessInNamespace(f.Client, podName, contName, f.Namespace.Name)
					Expect(err).NotTo(HaveOccurred())
				}
			})
		})
	})

	Describe("Pod containers", func() {
		Context("On scheduling a Guaranteed Pod", func() {
			It("Pod containers should have been created under the cgroup-root", func() {
				if framework.TestContext.CgroupsPerQOS {
					var guaranteedPod api.Pod
					By("Creating a Guaranteed pod in Namespace", func() {
						podName := "qos-pod" + string(uuid.NewUUID())
						contName := "qos-container" + string(uuid.NewUUID())
						pod := &api.Pod{
							ObjectMeta: api.ObjectMeta{
								Name:      podName,
								Namespace: f.Namespace.Name,
							},
							Spec: api.PodSpec{
								Containers: []api.Container{
									{
										Image:     framework.GetPauseImageName(f.Client),
										Name:      contName,
										Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
									},
								},
							},
						}
						podClient := f.PodClient()
						guaranteedPod := podClient.Create(pod)
					})
					By("Checking if the pod cgroup was created", func() {
						podName := "qos-pod" + string(uuid.NewUUID())
						contName := "qos-container" + string(uuid.NewUUID())
						pod := &api.Pod{
							ObjectMeta: api.ObjectMeta{
								Name:      podName,
								Namespace: f.Namespace.Name,
							},
							Spec: api.PodSpec{
								Containers: []api.Container{
									{
										Image:   "gcr.io/google_containers/busybox:1.24",
										Name:    contName,
										Command: []string{"sh", "-c", "if [ -d /tmp/memory/pod#" + guaranteedPod.UID + " ] && [ -d /tmp/cpu/pod#" + guaranteedPod.UID + " ]; then exit 0; else exit 1; fi"},
										VolumeMounts: []api.VolumeMount{
											{
												Name:      "sysfscgroup",
												MountPath: "/tmp",
											},
										},
									},
								},
								Volumes: []api.Volume{
									{
										Name: "sysfscgroup",
										VolumeSource: api.VolumeSource{
											HostPath: &api.HostPathVolumeSource{Path: "/sys/fs/cgroup"},
										},
									},
								},
							},
						}
						podClient := f.PodClient()
						podClient.Create(pod)
						err := framework.WaitForPodSuccessInNamespace(f.Client, podName, contName, f.Namespace.Name)
						Expect(err).NotTo(HaveOccurred())
					})
					By("Checking if the pod cgroup was deleted", func() {
						// delete the guaranteed pod.
						podUID := guaranteedPod.UID
						podClient := f.PodClient()
						Expect(podClient.Delete(guaranteedPod)).NotTo(HaveOccurred())
						// Now check that the cgroups get deleted eventually
						podName := "qos-pod" + string(uuid.NewUUID())
						contName := "qos-container" + string(uuid.NewUUID())
						pod := &api.Pod{
							ObjectMeta: api.ObjectMeta{
								Name:      podName,
								Namespace: f.Namespace.Name,
							},
							Spec: api.PodSpec{
								RestartPolicy: api.RestartPolicyOnFailure,
								Containers: []api.Container{
									{
										Image:   "gcr.io/google_containers/busybox:1.24",
										Name:    contName,
										Command: []string{"sh", "-c", "if [ -d /tmp/memory/pod#" + podUID + " ] || [ -d /tmp/cpu/pod#" + podUID + " ]; then exit 1; else exit 0; fi"},
										VolumeMounts: []api.VolumeMount{
											{
												Name:      "sysfscgroup",
												MountPath: "/tmp",
											},
										},
									},
								},
								Volumes: []api.Volume{
									{
										Name: "sysfscgroup",
										VolumeSource: api.VolumeSource{
											HostPath: &api.HostPathVolumeSource{Path: "/sys/fs/cgroup"},
										},
									},
								},
							},
						}
						podClient := f.PodClient()
						podClient.Create(pod)
						// This currently waits for five minutes before failing.
						err := framework.WaitForPodSuccessInNamespace(f.Client, podName, contName, f.Namespace.Name)
						Expect(err).NotTo(HaveOccurred())
					})
				}
			})

		})
		Context("On scheduling a BestEffort Pod", func() {
			It("Pod containers should have been created under the BestEffort cgroup", func() {
				if framework.TestContext.CgroupsPerQOS {
					var podUID string
					var bestEffortPod *api.Pod
					By("Creating a BestEffort pod in Namespace", func() {
						podName := "qos-pod" + string(uuid.NewUUID())
						contName := "qos-container" + string(uuid.NewUUID())
						pod := &api.Pod{
							ObjectMeta: api.ObjectMeta{
								Name:      podName,
								Namespace: f.Namespace.Name,
							},
							Spec: api.PodSpec{
								// Don't restart the Pod since it is expected to exit
								RestartPolicy: api.RestartPolicyNever,
								Containers: []api.Container{
									{
										Image:     framework.GetPauseImageName(f.Client),
										Name:      contName,
										Resources: getResourceRequirements(getResourceList("", ""), getResourceList("", "")),
									},
								},
							},
						}
						podClient := f.PodClient()
						bestEffortPod := podClient.Create(pod)
						podUID = string(apiPod.UID)
					})
					By("Checking if the pod cgroup was created", func() {
						podName := "qos-pod" + string(uuid.NewUUID())
						contName := "qos-container" + string(uuid.NewUUID())
						pod := &api.Pod{
							ObjectMeta: api.ObjectMeta{
								Name:      podName,
								Namespace: f.Namespace.Name,
							},
							Spec: api.PodSpec{
								// Don't restart the Pod since it is expected to exit
								RestartPolicy: api.RestartPolicyNever,
								Containers: []api.Container{
									{
										Image:   "gcr.io/google_containers/busybox:1.24",
										Name:    contName,
										Command: []string{"sh", "-c", "if [ -d /tmp/memory/BestEffort/pod#" + podUID + " ] && [ -d /tmp/cpu/BestEffort/pod#" + podUID + " ]; then exit 0; else exit 1; fi"},
										VolumeMounts: []api.VolumeMount{
											{
												Name:      "sysfscgroup",
												MountPath: "/tmp",
											},
										},
									},
								},
								Volumes: []api.Volume{
									{
										Name: "sysfscgroup",
										VolumeSource: api.VolumeSource{
											HostPath: &api.HostPathVolumeSource{Path: "/sys/fs/cgroup"},
										},
									},
								},
							},
						}
						podClient := f.PodClient()
						podClient.Create(pod)
						err := framework.WaitForPodSuccessInNamespace(f.Client, podName, contName, f.Namespace.Name)
						Expect(err).NotTo(HaveOccurred())
					})
					By("Checking if the pod cgroup was deleted", func() {
						// delete the best-effort pod.
						podClient := f.PodClient()
						Expect(podClient.Delete(bestEffortPod)).NotTo(HaveOccurred())
						// Now check that the cgroups get deleted eventually
						podName := "qos-pod" + string(uuid.NewUUID())
						contName := "qos-container" + string(uuid.NewUUID())
						pod := &api.Pod{
							ObjectMeta: api.ObjectMeta{
								Name:      podName,
								Namespace: f.Namespace.Name,
							},
							Spec: api.PodSpec{
								RestartPolicy: api.RestartPolicyOnFailure,
								Containers: []api.Container{
									{
										Image:   "gcr.io/google_containers/busybox:1.24",
										Name:    contName,
										Command: []string{"sh", "-c", "if [ -d /tmp/memory/BestEffort/pod#" + podUID + " ] || [ -d /tmp/cpu/BestEffort/pod#" + podUID + " ]; then exit 1; else exit 0; fi"},
										VolumeMounts: []api.VolumeMount{
											{
												Name:      "sysfscgroup",
												MountPath: "/tmp",
											},
										},
									},
								},
								Volumes: []api.Volume{
									{
										Name: "sysfscgroup",
										VolumeSource: api.VolumeSource{
											HostPath: &api.HostPathVolumeSource{Path: "/sys/fs/cgroup"},
										},
									},
								},
							},
						}
						podClient := f.PodClient()
						podClient.Create(pod)
						// This currently waits for five minutes before failing.
						err := framework.WaitForPodSuccessInNamespace(f.Client, podName, contName, f.Namespace.Name)
						Expect(err).NotTo(HaveOccurred())
					})

				}
			})
		})
	})
})

// getResourceList returns a ResourceList with the
// specified cpu and memory resource values
func getResourceList(cpu, memory string) api.ResourceList {
	res := api.ResourceList{}
	if cpu != "" {
		res[api.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[api.ResourceMemory] = resource.MustParse(memory)
	}
	return res
}

// getResourceRequirements returns a ResourceRequirements object
func getResourceRequirements(requests, limits api.ResourceList) api.ResourceRequirements {
	res := api.ResourceRequirements{}
	res.Requests = requests
	res.Limits = limits
	return res
}
