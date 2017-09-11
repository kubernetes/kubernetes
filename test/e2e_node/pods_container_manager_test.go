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
	"path"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// getResourceList returns a ResourceList with the
// specified cpu and memory resource values
func getResourceList(cpu, memory string) v1.ResourceList {
	res := v1.ResourceList{}
	if cpu != "" {
		res[v1.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[v1.ResourceMemory] = resource.MustParse(memory)
	}
	return res
}

// getResourceRequirements returns a ResourceRequirements object
func getResourceRequirements(requests, limits v1.ResourceList) v1.ResourceRequirements {
	res := v1.ResourceRequirements{}
	res.Requests = requests
	res.Limits = limits
	return res
}

const (
	// Kubelet internal cgroup name for node allocatable cgroup.
	defaultNodeAllocatableCgroup = "kubepods"
	// Kubelet internal cgroup name for burstable tier
	burstableCgroup = "burstable"
	// Kubelet internal cgroup name for burstable tier
	bestEffortCgroup = "burstable"
)

// makePodToVerifyCgroups returns a pod that verifies the existence of the specified cgroups.
func makePodToVerifyCgroups(cgroupNames []cm.CgroupName) *v1.Pod {
	// convert the names to their literal cgroupfs forms...
	cgroupFsNames := []string{}
	for _, cgroupName := range cgroupNames {
		// Add top level cgroup used to enforce node allocatable.
		cgroupName = cm.CgroupName(path.Join(defaultNodeAllocatableCgroup, string(cgroupName)))
		if framework.TestContext.KubeletConfig.CgroupDriver == "systemd" {
			cgroupFsNames = append(cgroupFsNames, cm.ConvertCgroupNameToSystemd(cgroupName, true))
		} else {
			cgroupFsNames = append(cgroupFsNames, string(cgroupName))
		}
	}
	glog.Infof("expecting %v cgroups to be found", cgroupFsNames)
	// build the pod command to either verify cgroups exist
	command := ""
	for _, cgroupFsName := range cgroupFsNames {
		localCommand := "if [ ! -d /tmp/memory/" + cgroupFsName + " ] || [ ! -d /tmp/cpu/" + cgroupFsName + " ]; then exit 1; fi; "
		command += localCommand
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Image:   busyboxImage,
					Name:    "container" + string(uuid.NewUUID()),
					Command: []string{"sh", "-c", command},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "sysfscgroup",
							MountPath: "/tmp",
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "sysfscgroup",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{Path: "/sys/fs/cgroup"},
					},
				},
			},
		},
	}
	return pod
}

// makePodToVerifyCgroupRemoved verfies the specified cgroup does not exist.
func makePodToVerifyCgroupRemoved(cgroupName cm.CgroupName) *v1.Pod {
	cgroupFsName := string(cgroupName)
	if framework.TestContext.KubeletConfig.CgroupDriver == "systemd" {
		cgroupFsName = cm.ConvertCgroupNameToSystemd(cm.CgroupName(cgroupName), true)
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyOnFailure,
			Containers: []v1.Container{
				{
					Image:   busyboxImage,
					Name:    "container" + string(uuid.NewUUID()),
					Command: []string{"sh", "-c", "for i in `seq 1 10`; do if [ ! -d /tmp/memory/" + cgroupFsName + " ] && [ ! -d /tmp/cpu/" + cgroupFsName + " ]; then exit 0; else sleep 10; fi; done; exit 1"},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "sysfscgroup",
							MountPath: "/tmp",
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "sysfscgroup",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{Path: "/sys/fs/cgroup"},
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
				if !framework.TestContext.KubeletConfig.CgroupsPerQOS {
					return
				}
				cgroupsToVerify := []cm.CgroupName{cm.CgroupName(burstableCgroup), cm.CgroupName(bestEffortCgroup)}
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
				if !framework.TestContext.KubeletConfig.CgroupsPerQOS {
					return
				}
				var (
					guaranteedPod *v1.Pod
					podUID        string
				)
				By("Creating a Guaranteed pod in Namespace", func() {
					guaranteedPod = f.PodClient().Create(&v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "pod" + string(uuid.NewUUID()),
							Namespace: f.Namespace.Name,
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
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
					Expect(f.PodClient().Delete(guaranteedPod.Name, &metav1.DeleteOptions{GracePeriodSeconds: &gp})).NotTo(HaveOccurred())
					pod := makePodToVerifyCgroupRemoved(cm.CgroupName("pod" + podUID))
					f.PodClient().Create(pod)
					err := framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
					Expect(err).NotTo(HaveOccurred())
				})
			})
		})
		Context("On scheduling a BestEffort Pod", func() {
			It("Pod containers should have been created under the BestEffort cgroup", func() {
				if !framework.TestContext.KubeletConfig.CgroupsPerQOS {
					return
				}
				var (
					podUID        string
					bestEffortPod *v1.Pod
				)
				By("Creating a BestEffort pod in Namespace", func() {
					bestEffortPod = f.PodClient().Create(&v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "pod" + string(uuid.NewUUID()),
							Namespace: f.Namespace.Name,
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
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
					cgroupsToVerify := []cm.CgroupName{cm.CgroupName("besteffort/pod" + podUID)}
					pod := makePodToVerifyCgroups(cgroupsToVerify)
					f.PodClient().Create(pod)
					err := framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
					Expect(err).NotTo(HaveOccurred())
				})
				By("Checking if the pod cgroup was deleted", func() {
					gp := int64(1)
					Expect(f.PodClient().Delete(bestEffortPod.Name, &metav1.DeleteOptions{GracePeriodSeconds: &gp})).NotTo(HaveOccurred())
					pod := makePodToVerifyCgroupRemoved(cm.CgroupName("besteffort/pod" + podUID))
					f.PodClient().Create(pod)
					err := framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
					Expect(err).NotTo(HaveOccurred())
				})
			})
		})
		Context("On scheduling a Burstable Pod", func() {
			It("Pod containers should have been created under the Burstable cgroup", func() {
				if !framework.TestContext.KubeletConfig.CgroupsPerQOS {
					return
				}
				var (
					podUID       string
					burstablePod *v1.Pod
				)
				By("Creating a Burstable pod in Namespace", func() {
					burstablePod = f.PodClient().Create(&v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "pod" + string(uuid.NewUUID()),
							Namespace: f.Namespace.Name,
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
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
					cgroupsToVerify := []cm.CgroupName{cm.CgroupName("burstable/pod" + podUID)}
					pod := makePodToVerifyCgroups(cgroupsToVerify)
					f.PodClient().Create(pod)
					err := framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
					Expect(err).NotTo(HaveOccurred())
				})
				By("Checking if the pod cgroup was deleted", func() {
					gp := int64(1)
					Expect(f.PodClient().Delete(burstablePod.Name, &metav1.DeleteOptions{GracePeriodSeconds: &gp})).NotTo(HaveOccurred())
					pod := makePodToVerifyCgroupRemoved(cm.CgroupName("burstable/pod" + podUID))
					f.PodClient().Create(pod)
					err := framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
					Expect(err).NotTo(HaveOccurred())
				})
			})
		})
	})
})
