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
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

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

// makePodToVerifyCgroups returns a pod that verifies the existence of the specified cgroups.
func makePodToVerifyCgroups(cgroupNames []cm.CgroupName) *api.Pod {
	// convert the names to their literal cgroupfs forms...
	cgroupFsNames := []string{}
	for _, cgroupName := range cgroupNames {
		if framework.TestContext.KubeletConfig.CgroupDriver == "systemd" {
			cgroupFsNames = append(cgroupFsNames, cm.ConvertCgroupNameToSystemd(cgroupName, true))
		} else {
			cgroupFsNames = append(cgroupFsNames, string(cgroupName))
		}
	}

	// build the pod command to either verify cgroups exist
	command := ""
	for _, cgroupFsName := range cgroupFsNames {
		localCommand := "if [ ! -d /tmp/memory/" + cgroupFsName + " ] || [ ! -d /tmp/cpu/" + cgroupFsName + " ]; then exit 1; fi; "
		command += localCommand
	}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "pod" + string(uuid.NewUUID()),
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyNever,
			Containers: []api.Container{
				{
					Image:   "gcr.io/google_containers/busybox:1.24",
					Name:    "container" + string(uuid.NewUUID()),
					Command: []string{"sh", "-c", command},
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
	return pod
}

// makePodToVerifyCgroupRemoved verfies the specified cgroup does not exist.
func makePodToVerifyCgroupRemoved(cgroupName cm.CgroupName) *api.Pod {
	cgroupFsName := string(cgroupName)
	if framework.TestContext.KubeletConfig.CgroupDriver == "systemd" {
		cgroupFsName = cm.ConvertCgroupNameToSystemd(cm.CgroupName(cgroupName), true)
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "pod" + string(uuid.NewUUID()),
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			Containers: []api.Container{
				{
					Image:   "gcr.io/google_containers/busybox:1.24",
					Name:    "container" + string(uuid.NewUUID()),
					Command: []string{"sh", "-c", "for i in `seq 1 10`; do if [ ! -d /tmp/memory/" + cgroupFsName + " ] && [ ! -d /tmp/cpu/" + cgroupFsName + " ]; then exit 0; else sleep 10; fi; done; exit 1"},
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
	return pod
}

var _ = framework.KubeDescribe("Kubelet Cgroup Manager", func() {
	f := framework.NewDefaultFramework("kubelet-cgroup-manager")
	Describe("QOS containers", func() {
		Context("On enabling QOS cgroup hierarchy", func() {
			It("Top level QoS containers should have been created", func() {
				if !framework.TestContext.KubeletConfig.ExperimentalCgroupsPerQOS {
					return
				}
				cgroupsToVerify := []cm.CgroupName{cm.CgroupName(qos.Burstable), cm.CgroupName(qos.BestEffort)}
				pod := makePodToVerifyCgroups(cgroupsToVerify)
				f.PodClient().Create(pod)
				err := framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
				Expect(err).NotTo(HaveOccurred())
			})
		})
	})

	Describe("Pod containers", func() {
		Context("On scheduling a Guaranteed Pod", func() {
			It("Pod containers should have been created under the cgroup-root", func() {
				if !framework.TestContext.KubeletConfig.ExperimentalCgroupsPerQOS {
					return
				}
				var (
					guaranteedPod *api.Pod
					podUID        string
				)
				By("Creating a Guaranteed pod in Namespace", func() {
					guaranteedPod = f.PodClient().Create(&api.Pod{
						ObjectMeta: api.ObjectMeta{
							Name:      "pod" + string(uuid.NewUUID()),
							Namespace: f.Namespace.Name,
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Image:     framework.GetPauseImageName(f.ClientSet),
									Name:      "container" + string(uuid.NewUUID()),
									Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
								},
							},
						},
					})
					podUID = string(guaranteedPod.UID)
				})
				By("Checking if the pod cgroup was created", func() {
					cgroupsToVerify := []cm.CgroupName{cm.CgroupName("pod" + podUID)}
					pod := makePodToVerifyCgroups(cgroupsToVerify)
					f.PodClient().Create(pod)
					err := framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
					Expect(err).NotTo(HaveOccurred())
				})
				By("Checking if the pod cgroup was deleted", func() {
					gp := int64(1)
					Expect(f.PodClient().Delete(guaranteedPod.Name, &api.DeleteOptions{GracePeriodSeconds: &gp})).NotTo(HaveOccurred())
					pod := makePodToVerifyCgroupRemoved(cm.CgroupName("pod" + podUID))
					f.PodClient().Create(pod)
					err := framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
					Expect(err).NotTo(HaveOccurred())
				})
			})
		})
		Context("On scheduling a BestEffort Pod", func() {
			It("Pod containers should have been created under the BestEffort cgroup", func() {
				if !framework.TestContext.KubeletConfig.ExperimentalCgroupsPerQOS {
					return
				}
				var (
					podUID        string
					bestEffortPod *api.Pod
				)
				By("Creating a BestEffort pod in Namespace", func() {
					bestEffortPod = f.PodClient().Create(&api.Pod{
						ObjectMeta: api.ObjectMeta{
							Name:      "pod" + string(uuid.NewUUID()),
							Namespace: f.Namespace.Name,
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Image:     framework.GetPauseImageName(f.ClientSet),
									Name:      "container" + string(uuid.NewUUID()),
									Resources: getResourceRequirements(getResourceList("", ""), getResourceList("", "")),
								},
							},
						},
					})
					podUID = string(bestEffortPod.UID)
				})
				By("Checking if the pod cgroup was created", func() {
					cgroupsToVerify := []cm.CgroupName{cm.CgroupName("BestEffort/pod" + podUID)}
					pod := makePodToVerifyCgroups(cgroupsToVerify)
					f.PodClient().Create(pod)
					err := framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
					Expect(err).NotTo(HaveOccurred())
				})
				By("Checking if the pod cgroup was deleted", func() {
					gp := int64(1)
					Expect(f.PodClient().Delete(bestEffortPod.Name, &api.DeleteOptions{GracePeriodSeconds: &gp})).NotTo(HaveOccurred())
					pod := makePodToVerifyCgroupRemoved(cm.CgroupName("BestEffort/pod" + podUID))
					f.PodClient().Create(pod)
					err := framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
					Expect(err).NotTo(HaveOccurred())
				})
			})
		})
		Context("On scheduling a Burstable Pod", func() {
			It("Pod containers should have been created under the Burstable cgroup", func() {
				if !framework.TestContext.KubeletConfig.ExperimentalCgroupsPerQOS {
					return
				}
				var (
					podUID       string
					burstablePod *api.Pod
				)
				By("Creating a Burstable pod in Namespace", func() {
					burstablePod = f.PodClient().Create(&api.Pod{
						ObjectMeta: api.ObjectMeta{
							Name:      "pod" + string(uuid.NewUUID()),
							Namespace: f.Namespace.Name,
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Image:     framework.GetPauseImageName(f.ClientSet),
									Name:      "container" + string(uuid.NewUUID()),
									Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi")),
								},
							},
						},
					})
					podUID = string(burstablePod.UID)
				})
				By("Checking if the pod cgroup was created", func() {
					cgroupsToVerify := []cm.CgroupName{cm.CgroupName("Burstable/pod" + podUID)}
					pod := makePodToVerifyCgroups(cgroupsToVerify)
					f.PodClient().Create(pod)
					err := framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
					Expect(err).NotTo(HaveOccurred())
				})
				By("Checking if the pod cgroup was deleted", func() {
					gp := int64(1)
					Expect(f.PodClient().Delete(burstablePod.Name, &api.DeleteOptions{GracePeriodSeconds: &gp})).NotTo(HaveOccurred())
					pod := makePodToVerifyCgroupRemoved(cm.CgroupName("Burstable/pod" + podUID))
					f.PodClient().Create(pod)
					err := framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
					Expect(err).NotTo(HaveOccurred())
				})
			})
		})
	})
})
