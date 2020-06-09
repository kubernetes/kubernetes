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

package e2enode

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"

	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

// makePodToVerifyPids returns a pod that verifies specified cgroup with pids
func makePodToVerifyPids(baseName string, pidsLimit resource.Quantity) *v1.Pod {
	// convert the cgroup name to its literal form
	cgroupFsName := ""
	cgroupName := cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup, baseName)
	if framework.TestContext.KubeletConfig.CgroupDriver == "systemd" {
		cgroupFsName = cgroupName.ToSystemd()
	} else {
		cgroupFsName = cgroupName.ToCgroupfs()
	}

	// this command takes the expected value and compares it against the actual value for the pod cgroup pids.max
	command := ""
	if IsCgroup2UnifiedMode() {
		command = fmt.Sprintf("expected=%v; actual=$(cat /tmp/%v/pids.max); if [ \"$expected\" -ne \"$actual\" ]; then exit 1; fi; ", pidsLimit.Value(), cgroupFsName)
	} else {
		command = fmt.Sprintf("expected=%v; actual=$(cat /tmp/pids/%v/pids.max); if [ \"$expected\" -ne \"$actual\" ]; then exit 1; fi; ", pidsLimit.Value(), cgroupFsName)
	}

	framework.Logf("Pod to run command: %v", command)
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

func runPodPidsLimitTests(f *framework.Framework) {
	ginkgo.It("should set pids.max for Pod", func() {
		ginkgo.By("by creating a G pod")
		pod := f.PodClient().Create(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Image: imageutils.GetPauseImageName(),
						Name:  "container" + string(uuid.NewUUID()),
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceName("cpu"):    resource.MustParse("10m"),
								v1.ResourceName("memory"): resource.MustParse("100Mi"),
							},
						},
					},
				},
			},
		})
		podUID := string(pod.UID)
		ginkgo.By("checking if the expected pids settings were applied")
		verifyPod := makePodToVerifyPids("pod"+podUID, resource.MustParse("1024"))
		f.PodClient().Create(verifyPod)
		err := e2epod.WaitForPodSuccessInNamespace(f.ClientSet, verifyPod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)
	})
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("PodPidsLimit [Serial] [Feature:SupportPodPidsLimit][NodeFeature:SupportPodPidsLimit]", func() {
	f := framework.NewDefaultFramework("pids-limit-test")
	ginkgo.Context("With config updated with pids feature enabled", func() {
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
