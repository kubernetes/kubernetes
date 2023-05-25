/*
Copyright 2022 The Kubernetes Authors.

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
	"net/http"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

// Pod Termination State Test will cover both pod termination with graceful
// shutdown and oomkilled cases, in both cases, it will verify
// 1. pod is in termating status after receiving sig term
// 2. cgroups associated with the pod are deleted
// 3. pod is no longer existed in API server after termination
var _ = SIGDescribe("Pod Termination State [NodeConformance]", func() {
	f := framework.NewDefaultFramework("pod-termination-state-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	// Normal Pod Cases
	normalContainerName := "normal-container"
	podSpec := getNormalPod("normal-pod", normalContainerName)
	runNormalTest(f, podSpec, normalContainerName)

	// Evicted Pod Cases (oomkill)
	containerName := "oomkill-target-container"
	oomPodSpec := getOOMTargetPod("oomkill-target-pod", containerName)
	runEvictedTest(f, oomPodSpec, containerName)
})

func runNormalTest(f *framework.Framework, podSpec *v1.Pod, containerName string) {
	ginkgo.Context("", func() {
		var podClient *e2epod.PodClient
		ginkgo.BeforeEach(func() {
			ginkgo.By("setting up the pod to be used in the test")
			podClient = e2epod.NewPodClient(f)
			podClient.Create(context.TODO(), podSpec)
		})

		ginkgo.It("The pod should be running and deleted with grace period", func() {

			// We need to wait for the pod to be running, otherwise the deletion
			// may be carried out immediately rather than gracefully.
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, f.Namespace.Name))

			// save the running pod
			pod, err := podClient.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to GET scheduled pod")

			// save the UID for cgroup verify
			podUID := string(pod.UID)

			// delete pod gracefully
			ginkgo.By("deleting the pod gracefully")
			var lastPod v1.Pod
			var statusCode int
			err = f.ClientSet.CoreV1().RESTClient().Delete().AbsPath("/api/v1/namespaces", pod.Namespace, "pods", pod.Name).Param("gracePeriodSeconds", "30").Do(context.TODO()).StatusCode(&statusCode).Into(&lastPod)
			framework.ExpectNoError(err, "failed to use http client to send delete")
			framework.ExpectEqual(statusCode, http.StatusOK, "failed to delete gracefully by client request")

			// verifying the kubelet observed the termination notice
			// check pod status is from running to terminating status, after a while, pod is deleted in kubelet
			err = e2epod.WaitForPodTerminatingInNamespaceTimeout(context.TODO(), f.ClientSet, podSpec.Name, f.Namespace.Name, 30)
			framework.ExpectNoError(err)

			// Sleep enough time to ensure the pod is terminated
			time.Sleep(30 * time.Second)

			// resources are cleaned up and phase is set correctly cgroup files
			// see same way in pods_container_manager_test.go
			new_pod := makePodToVerifyCgroupRemoved("pod" + podUID)
			podClient.Create(context.TODO(), new_pod)
			err = e2epod.WaitForPodSuccessInNamespace(context.TODO(), f.ClientSet, new_pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)

			// verify pod is deleted from API server
			ginkgo.By("Fetching the latest pod status")
			err = e2epod.WaitForPodNotFoundInNamespace(context.TODO(), f.ClientSet, podSpec.Name, f.Namespace.Name, 120)
			framework.ExpectNoError(err)
		})

		ginkgo.AfterEach(func() {
			ginkgo.By(fmt.Sprintf("deleting pod: %s", podSpec.Name))
			e2epod.NewPodClient(f).DeleteSync(context.TODO(), podSpec.Name, metav1.DeleteOptions{}, framework.PodDeleteTimeout)
		})
	})
}

func runEvictedTest(f *framework.Framework, podSpec *v1.Pod, containerName string) {
	ginkgo.Context("", func() {
		var podClient *e2epod.PodClient
		ginkgo.BeforeEach(func() {
			ginkgo.By("setting up the pod to be used in the test")
			podClient = e2epod.NewPodClient(f)
			podClient.Create(context.TODO(), podSpec)
		})

		ginkgo.It("The containers terminated by OOM killer should have the reason set to OOMKilled", func() {
			ginkgo.By("Waiting for the pod to be failed")
			e2epod.WaitForPodTerminatedInNamespace(context.TODO(), f.ClientSet, podSpec.Name, "", f.Namespace.Name)

			ginkgo.By("Fetching the latest pod status")
			pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "Failed to get the recent pod object for name: %q", pod.Name)

			ginkgo.By("Verifying the OOM target container has the expected reason")
			verifyReasonForOOMKilledContainer(pod, containerName)

			podUID := string(pod.UID)
			// resources are cleaned up and phase is set correctly cgroup files
			// see example in pods_container_manager_test.go
			new_pod := makePodToVerifyCgroupRemoved("pod" + podUID)
			podClient.Create(context.TODO(), new_pod)
			err = e2epod.WaitForPodSuccessInNamespace(context.TODO(), f.ClientSet, new_pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		ginkgo.AfterEach(func() {
			ginkgo.By(fmt.Sprintf("deleting pod: %s", podSpec.Name))
			e2epod.NewPodClient(f).DeleteSync(context.TODO(), podSpec.Name, metav1.DeleteOptions{}, framework.PodDeleteTimeout)
		})
	})
}

func getNormalPod(podName string, ctnName string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				getNormalContainer(ctnName),
			},
		},
	}
}

func getNormalContainer(name string) v1.Container {
	return v1.Container{
		Name:  name,
		Image: busyboxImage,
		Command: []string{
			"sh",
			"-c",
			// do normal staff
			"sleep 5 && echo 'Hello'",
		},
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("15Mi"),
			},
			Limits: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("15Mi"),
			},
		},
	}
}
