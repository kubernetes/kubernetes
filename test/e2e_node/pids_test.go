/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"time"

	apiv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"

	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// makePodToVerifyPids returns a pod that verifies specified cgroup with pids
func makePodToVerifyPids(baseName string, pidsLimit resource.Quantity) *apiv1.Pod {
	// convert the cgroup name to its literal form
	cgroupFsName := ""
	cgroupName := cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup, baseName)
	if framework.TestContext.KubeletConfig.CgroupDriver == "systemd" {
		cgroupFsName = cgroupName.ToSystemd()
	} else {
		cgroupFsName = cgroupName.ToCgroupfs()
	}

	// this command takes the expected value and compares it against the actual value for the pod cgroup pids.max
	command := fmt.Sprintf("expected=%v; actual=$(cat /tmp/pids/%v/pids.max); if [ \"$expected\" -ne \"$actual\" ]; then exit 1; fi; ", pidsLimit.Value(), cgroupFsName)
	framework.Logf("Pod to run command: %v", command)
	pod := &apiv1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod" + string(uuid.NewUUID()),
		},
		Spec: apiv1.PodSpec{
			RestartPolicy: apiv1.RestartPolicyNever,
			Containers: []apiv1.Container{
				{
					Image:   busyboxImage,
					Name:    "container" + string(uuid.NewUUID()),
					Command: []string{"sh", "-c", command},
					VolumeMounts: []apiv1.VolumeMount{
						{
							Name:      "sysfscgroup",
							MountPath: "/tmp",
						},
					},
				},
			},
			Volumes: []apiv1.Volume{
				{
					Name: "sysfscgroup",
					VolumeSource: apiv1.VolumeSource{
						HostPath: &apiv1.HostPathVolumeSource{Path: "/sys/fs/cgroup"},
					},
				},
			},
		},
	}
	return pod
}

// enablePodPidsLimitInKubelet enables pod pid limit feature for kubelet with a sensible default test limit
func enablePodPidsLimitInKubelet(f *framework.Framework) *kubeletconfig.KubeletConfiguration {
	oldCfg, err := getCurrentKubeletConfig()
	framework.ExpectNoError(err)
	newCfg := oldCfg.DeepCopy()
	if newCfg.FeatureGates == nil {
		newCfg.FeatureGates = make(map[string]bool)
		newCfg.FeatureGates["SupportPodPidsLimit"] = true
	}
	newCfg.PodPidsLimit = int64(1024)
	// Update the Kubelet configuration.
	framework.ExpectNoError(setKubeletConfiguration(f, newCfg))

	// Wait for the Kubelet to be ready.
	Eventually(func() bool {
		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		return len(nodeList.Items) == 1
	}, time.Minute, time.Second).Should(BeTrue())

	return oldCfg
}

func runPodPidsLimitTests(f *framework.Framework) {
	It("should set pids.max for Pod", func() {
		By("by creating a G pod")
		pod := f.PodClient().Create(&apiv1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: apiv1.PodSpec{
				Containers: []apiv1.Container{
					{
						Image: imageutils.GetPauseImageName(),
						Name:  "container" + string(uuid.NewUUID()),
						Resources: apiv1.ResourceRequirements{
							Limits: apiv1.ResourceList{
								apiv1.ResourceName("cpu"):    resource.MustParse("10m"),
								apiv1.ResourceName("memory"): resource.MustParse("100Mi"),
							},
						},
					},
				},
			},
		})
		podUID := string(pod.UID)
		By("checking if the expected pids settings were applied")
		verifyPod := makePodToVerifyPids("pod"+podUID, resource.MustParse("1024"))
		f.PodClient().Create(verifyPod)
		err := framework.WaitForPodSuccessInNamespace(f.ClientSet, verifyPod.Name, f.Namespace.Name)
		Expect(err).NotTo(HaveOccurred())
	})
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("PodPidsLimit [Serial] [Feature:SupportPodPidsLimit][NodeFeature:SupportPodPidsLimit]", func() {
	f := framework.NewDefaultFramework("pids-limit-test")
	Context("With config updated with pids feature enabled", func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
			initialConfig.FeatureGates["SupportPodPidsLimit"] = true
			initialConfig.PodPidsLimit = int64(1024)
		})
		runPodPidsLimitTests(f)
	})
})
