/*
Copyright 2023 The Kubernetes Authors.

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
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Pod Termination", func() {
	f := framework.NewDefaultFramework("containers-lifecycle-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	ginkgo.When("a pod cannot terminate gracefully", func() {
		testPod := func(name string, gracePeriod int64) *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "busybox",
							Image: imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{
								"sleep",
								"10000",
							},
						},
					},
					TerminationGracePeriodSeconds: &gracePeriod,
				},
			}
		}

		// To account for the time it takes to delete the pod, we add a 10 second
		// buffer. The 10 second buffer is arbitrary, but it should be enough to
		// account for the time it takes to delete the pod.
		bufferSeconds := int64(10)

		ginkgo.It("should respect termination grace period seconds [NodeConformance]", func() {
			client := e2epod.NewPodClient(f)
			gracePeriod := int64(30)

			ginkgo.By("creating a pod with a termination grace period seconds")
			pod := testPod("pod-termination-grace-period", gracePeriod)
			pod = client.Create(context.TODO(), pod)

			ginkgo.By("ensuring the pod is running")
			err := e2epod.WaitForPodRunningInNamespace(context.TODO(), f.ClientSet, pod)
			framework.ExpectNoError(err)

			ginkgo.By("deleting the pod gracefully")
			err = client.Delete(context.TODO(), pod.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("ensuring the pod is terminated within the grace period seconds + buffer seconds")
			err = e2epod.WaitForPodNotFoundInNamespace(context.TODO(), f.ClientSet, pod.Name, pod.Namespace, time.Duration(gracePeriod+bufferSeconds)*time.Second)
			framework.ExpectNoError(err)
		})

		ginkgo.It("should respect termination grace period seconds with long-running preStop hook [NodeConformance]", func() {
			client := e2epod.NewPodClient(f)
			gracePeriod := int64(30)

			ginkgo.By("creating a pod with a termination grace period seconds and long-running preStop hook")
			pod := testPod("pod-termination-grace-period", gracePeriod)
			pod.Spec.Containers[0].Lifecycle = &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					Exec: &v1.ExecAction{
						Command: []string{
							"sleep",
							"10000",
						},
					},
				},
			}
			pod = client.Create(context.TODO(), pod)

			ginkgo.By("ensuring the pod is running")
			err := e2epod.WaitForPodRunningInNamespace(context.TODO(), f.ClientSet, pod)
			framework.ExpectNoError(err)

			ginkgo.By("deleting the pod gracefully")
			err = client.Delete(context.TODO(), pod.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("ensuring the pod is terminated within the grace period seconds + buffer seconds")
			err = e2epod.WaitForPodNotFoundInNamespace(context.TODO(), f.ClientSet, pod.Name, pod.Namespace, time.Duration(gracePeriod+bufferSeconds)*time.Second)
			framework.ExpectNoError(err)
		})
	})
})
