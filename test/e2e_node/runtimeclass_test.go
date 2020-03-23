/*
Copyright 2020 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"path/filepath"
	"strings"

	"k8s.io/api/core/v1"
	nodev1beta1 "k8s.io/api/node/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

// makePodToVerifyCgroups returns a pod that verifies the existence of the specified cgroups.
func makePodToVerifyCgroupSize(cgroupNames []string, expectedCPU string, expectedMemory string) *v1.Pod {
	// convert the names to their literal cgroupfs forms...
	cgroupFsNames := []string{}
	rootCgroupName := cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup)
	for _, baseName := range cgroupNames {
		// Add top level cgroup used to enforce node allocatable.
		cgroupComponents := strings.Split(baseName, "/")
		cgroupName := cm.NewCgroupName(rootCgroupName, cgroupComponents...)
		cgroupFsNames = append(cgroupFsNames, toCgroupFsName(cgroupName))
	}
	framework.Logf("expecting %v cgroups to be found", cgroupFsNames)

	// build the pod command to verify cgroup sizing
	command := ""
	for _, cgroupFsName := range cgroupFsNames {
		memLimitCgroup := filepath.Join("/host_cgroups/memory", cgroupFsName, "memory.limit_in_bytes")
		cpuQuotaCgroup := filepath.Join("/host_cgroups/cpu", cgroupFsName, "cpu.cfs_quota_us")
		localCommand := "if [ ! $(cat " + memLimitCgroup + ") == " + expectedMemory + " ] || [ ! $(cat " + cpuQuotaCgroup + ") == " + expectedCPU + " ]; then exit 1; fi; "

		framework.Logf("command: %v: ", localCommand)
		command += localCommand
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "cgroup-verification-pod-",
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Image:   busyboxImage,
					Name:    "container",
					Command: []string{"sh", "-c", command},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "sysfscgroup",
							MountPath: "/host_cgroups",
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

var _ = framework.KubeDescribe("Kubelet PodOverhead handling [LinuxOnly]", func() {
	f := framework.NewDefaultFramework("podoverhead-handling")
	ginkgo.Describe("PodOverhead cgroup accounting", func() {
		ginkgo.Context("On running pod with PodOverhead defined", func() {
			ginkgo.It("Pod cgroup should be sum of overhead and resource limits", func() {
				if !framework.TestContext.KubeletConfig.CgroupsPerQOS {
					return
				}

				var (
					guaranteedPod *v1.Pod
					podUID        string
					handler       string
				)
				ginkgo.By("Creating a RuntimeClass with Overhead definied", func() {
					handler = e2enode.PreconfiguredRuntimeClassHandler(framework.TestContext.ContainerRuntime)
					rc := &nodev1beta1.RuntimeClass{
						ObjectMeta: metav1.ObjectMeta{Name: handler},
						Handler:    handler,
						Overhead: &nodev1beta1.Overhead{
							PodFixed: getResourceList("200m", "140Mi"),
						},
					}
					_, err := f.ClientSet.NodeV1beta1().RuntimeClasses().Create(context.TODO(), rc, metav1.CreateOptions{})
					framework.ExpectNoError(err, "failed to create RuntimeClass resource")
				})
				ginkgo.By("Creating a Guaranteed pod with which has Overhead defined", func() {
					guaranteedPod = f.PodClient().CreateSync(&v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							GenerateName: "pod-with-overhead-",
							Namespace:    f.Namespace.Name,
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Image:     imageutils.GetPauseImageName(),
									Name:      "container",
									Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
								},
							},
							RuntimeClassName: &handler,
							Overhead:         getResourceList("200m", "140Mi"),
						},
					})
					podUID = string(guaranteedPod.UID)
				})
				ginkgo.By("Checking if the pod cgroup was created appropriately", func() {
					cgroupsToVerify := []string{"pod" + podUID}
					pod := makePodToVerifyCgroupSize(cgroupsToVerify, "30000", "251658240")
					pod = f.PodClient().Create(pod)
					err := e2epod.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
					framework.ExpectNoError(err)
				})
			})
		})
	})
})
