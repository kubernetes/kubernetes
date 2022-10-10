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

package e2enode

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

type testCase struct {
	podSpec                    *v1.Pod
	oomTargetContainerName     string
	wantPodDisruptionCondition *v1.PodCondition
}

var _ = SIGDescribe("OOMKiller [LinuxOnly]", func() {
	f := framework.NewDefaultFramework("oomkiller-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	ginkgo.Context("Pod disruption conditions enabled [Slow] [Serial] [Disruptive] [Feature:PodDisruptionConditions]", func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FeatureGates = map[string]bool{
				string(features.PodDisruptionConditions): true,
			}
		})

		testCases := []testCase{
			{
				podSpec: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod-oomkill-target-pod",
					},
					Spec: v1.PodSpec{
						RestartPolicy: v1.RestartPolicyNever,
						Containers:    []v1.Container{getOOMTargetContainer("pod-oomkill-target-container")},
					},
				},
				oomTargetContainerName: "pod-oomkill-target-container",
				wantPodDisruptionCondition: &v1.PodCondition{
					Type:    ResourceExhausted,
					Status:  v1.ConditionTrue,
					Reason:  "OOMKilled",
					Message: "OOMKilled container: pod-oomkill-target-container",
				},
			},
			{
				podSpec: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod-oomkill-with-init-oomtarget-pod",
					},
					Spec: v1.PodSpec{
						RestartPolicy:  v1.RestartPolicyNever,
						InitContainers: []v1.Container{getOOMTargetContainer("pod-oomkill-with-init-oomtarget-container")},
						Containers:     []v1.Container{getInnocentContainer("pod-oomkill-with-init-innocent-container")},
					},
				},
				oomTargetContainerName: "pod-oomkill-with-init-oomtarget-container",
				wantPodDisruptionCondition: &v1.PodCondition{
					Type:    ResourceExhausted,
					Status:  v1.ConditionTrue,
					Reason:  "OOMKilled",
					Message: "OOMKilled container: pod-oomkill-with-init-oomtarget-container",
				},
			},
		}
		runOomKillerTest(f, testCases)
	})

})

func runOomKillerTest(f *framework.Framework, testCases []testCase) {
	ginkgo.Context("", func() {

		ginkgo.BeforeEach(func() {
			for _, testCase := range testCases {
				ginkgo.By("setting up the pod to be used in the test")
				e2epod.NewPodClient(f).Create(testCase.podSpec)
			}
		})

		ginkgo.It("The containers terminated by OOM killer should have the reason set to OOMKilled", func() {

			for _, testCase := range testCases {
				ginkgo.By("Waiting for the pod to be failed")
				e2epod.WaitForPodTerminatedInNamespace(f.ClientSet, testCase.podSpec.Name, "", f.Namespace.Name)

				ginkgo.By("Fetching the latest pod status")
				pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), testCase.podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "Failed to get the recent pod object for name: %q", pod.Name)

				ginkgo.By("Verifying the OOM target container has the expected reason")
				verifyReasonForOOMKilledContainer(pod, testCase.oomTargetContainerName)

				if testCase.wantPodDisruptionCondition != nil {
					ginkgo.By("Verifying the pod has the expected pod condition")
					e2epod.VerifyPodHasCondition(f, pod, *testCase.wantPodDisruptionCondition)
				}
			}
		})

		ginkgo.AfterEach(func() {
			for _, testCase := range testCases {
				ginkgo.By(fmt.Sprintf("deleting pod: %s", testCase.podSpec.Name))
				e2epod.NewPodClient(f).DeleteSync(testCase.podSpec.Name, metav1.DeleteOptions{}, e2epod.PodDeleteTimeout)
			}
		})
	})
}

func verifyReasonForOOMKilledContainer(pod *v1.Pod, oomTargetContainerName string) {
	container := e2epod.FindContainerStatusInPod(pod, oomTargetContainerName)
	if container == nil {
		framework.Failf("OOM target pod %q, container %q does not have the expected state terminated", pod.Name, container.Name)
	}
	if container.State.Terminated == nil {
		framework.Failf("OOM target pod %q, container %q is not in the terminated state", pod.Name, container.Name)
	}
	framework.ExpectEqual(container.State.Terminated.Reason, "OOMKilled",
		fmt.Sprintf("pod: %q, container: %q has unexpected reason: %q", pod.Name, container.Name, container.State.Terminated.Reason))
}

func getOOMTargetContainer(name string) v1.Container {
	return v1.Container{
		Name:  name,
		Image: busyboxImage,
		Command: []string{
			"sh",
			"-c",
			// the script iterates allocation of variables with random values
			fmt.Sprintf("i=0; while [ $i -lt %d ]; do %s i=$(($i+1)); done", 10000000, "eval array$i=$RANDOM"),
		},
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("20Mi"),
			},
			Limits: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("20Mi"),
			},
		},
	}
}

func getInnocentContainer(name string) v1.Container {
	return v1.Container{
		Image: busyboxImage,
		Name:  name,
		Command: []string{
			"sh",
			"-c",
			"echo hello",
		},
	}
}
