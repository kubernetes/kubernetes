/*
Copyright 2025 The Kubernetes Authors.

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
	"strconv"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/common/node/framework/cgroups"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

// DISCLAIMER: I have not verified that any part of this test works.
// This is a test written entirely on high-level, theoretical knowledge.
// Consumers will more than likely need to tweak this code.
//
// TODO: May need other checks for node, windows, and cgroup2.
var _ = SIGDescribe("Pod-level Resources Feature Gate", framework.WithSerial(), ginkgo.Ordered, feature.PodLevelResources, framework.WithFeatureGate(features.PodLevelResources), func() {
	f := framework.NewDefaultFramework("pod-level-resources-feature-gate")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var podClient *pod.PodClient
	var normalPod, disabledPod *v1.Pod
	var podResources *cgroups.ContainerResources
	ginkgo.BeforeEach(func() {
		podClient = pod.NewPodClient(f)
		podResources = &cgroups.ContainerResources{
			CPUReq: "100m",
			CPULim: "100m",
			MemReq: "100Mi",
			MemLim: "100Mi",
		}
		normalPod = singleContainerPodWithNameAndResources(f, "happy-path", podResources)
		disabledPod = singleContainerPodWithNameAndResources(f, "rejected-when-disabled", podResources)
	})
	ginkgo.Context("Enable pod-level resources", func() {
		ginkgo.It("Create pod with pod-level resource specification", func(ctx context.Context) {
			gotNormalPod := podClient.CreateSync(ctx, normalPod)
			verifySingleContainerPod(ctx, f, podResources, gotNormalPod)
		})
	})
	ginkgo.Context("Disable pod-level resources", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *config.KubeletConfiguration) {
			initialConfig.FeatureGates = map[string]bool{"PodLevelResources": false}
		})
		ginkgo.It("New pod with pod-level resources rejected", func(ctx context.Context) {
			gotDisabledPod := podClient.Create(ctx, disabledPod)
			// TODO(132446): Once the blockers for this issue are fixed,
			// determine the failure reason from testing, either manually or by running this test.
			err := pod.WaitForPodContainerToFail(ctx, f.ClientSet, f.Namespace.Name, gotDisabledPod.Name, 0, "CrashLoopBackOff", 1*time.Minute)
			gomega.Expect(err).To(gomega.HaveOccurred())

			gotNormalPod, err := podClient.Get(ctx, normalPod.Name, metav1.GetOptions{
				TypeMeta: normalPod.TypeMeta,
			})
			framework.ExpectNoError(err, "failed to get existing pod with pod-level resources: %s", err)
			verifySingleContainerPod(ctx, f, podResources, gotNormalPod)
		})
	})
	ginkgo.Context("Pod-level resources re-enabled", func() {
		ginkgo.It("All pods running", func(ctx context.Context) {
			err := pod.WaitForPodRunningInNamespace(ctx, f.ClientSet, disabledPod)
			framework.ExpectNoError(err, "pod not running even after enabling pod-level resources: %s", err)
			gotDisabledPod, err := podClient.Get(ctx, disabledPod.Name, metav1.GetOptions{
				TypeMeta: disabledPod.TypeMeta,
			})
			framework.ExpectNoError(err, "failed to get existing pod with pod-level resources: %s", err)
			verifySingleContainerPod(ctx, f, podResources, gotDisabledPod)

			gotNormalPod, err := podClient.Get(ctx, normalPod.Name, metav1.GetOptions{
				TypeMeta: normalPod.TypeMeta,
			})
			framework.ExpectNoError(err, "failed to get existing pod with pod-level resources: %s", err)
			verifySingleContainerPod(ctx, f, podResources, gotNormalPod)
		})
	})
	ginkgo.AfterAll(func() {
		ctx := context.Background()
		err := pod.DeletePodWithWait(ctx, f.ClientSet, normalPod)
		framework.ExpectNoError(err, "failed to delete pod %s", err)
		err = pod.DeletePodWithWait(ctx, f.ClientSet, disabledPod)
		framework.ExpectNoError(err, "failed to delete pod %s", err)
	})
})

func singleContainerPodWithNameAndResources(f *framework.Framework, name string, podResources *cgroups.ContainerResources) *v1.Pod {
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: f.Namespace.Name,
			Labels:    map[string]string{"time": strconv.Itoa(time.Now().Nanosecond())},
		},
		Spec: v1.PodSpec{
			Resources: podResources.ResourceRequirements(),
			Containers: []v1.Container{
				cgroups.MakeContainerWithResources("c1", podResources, pod.InfiniteSleepCommand),
			},
		},
	}
	cgroups.ConfigureHostPathForPodCgroup(testPod)
	return testPod
}

func verifySingleContainerPod(ctx context.Context, f *framework.Framework, podResources *cgroups.ContainerResources, gotPod *v1.Pod) {
	isReady := pod.CheckPodsRunningReady(ctx, f.ClientSet, gotPod.Namespace, []string{
		gotPod.Name,
	}, 1*time.Minute)
	gomega.Expect(isReady).To(gomega.BeTrue())

	requirements := podResources.ResourceRequirements()
	gomega.Expect(gotPod.Spec.Resources).To(gomega.Equal(requirements))
	err := cgroups.VerifyPodCgroups(ctx, f, gotPod, podResources)
	framework.ExpectNoError(err, "failed to verify pod's cgroup values: %v", err)

	gomega.Expect(gotPod.Spec.Containers).To(gomega.HaveLen(1))
	gomega.Expect(gotPod.Spec.Containers[0].Resources).To(gomega.Equal(requirements))
	containerName := gotPod.Spec.Containers[0].Name
	err = cgroups.VerifyContainerMemoryLimit(f, gotPod, containerName, requirements, true)
	framework.ExpectNoError(err, "failed to verify memory limit cgroup value: %v", err)
	err = cgroups.VerifyContainerCPULimit(f, gotPod, containerName, requirements, true)
	framework.ExpectNoError(err, "failed to verify cpu limit cgroup value: %v", err)
}
