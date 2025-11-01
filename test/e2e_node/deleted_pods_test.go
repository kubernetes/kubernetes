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
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	testFinalizer = "example.com/test-finalizer"
)

var _ = SIGDescribe("Deleted pods handling", framework.WithNodeConformance(), func() {
	f := framework.NewDefaultFramework("deleted-pods-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.It("Should transition to Failed phase a pod which is deleted while pending", func(ctx context.Context) {
		podName := "deleted-pending-" + string(uuid.NewUUID())
		podSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:       podName,
				Finalizers: []string{testFinalizer},
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyAlways,
				Containers: []v1.Container{
					{
						Name:            podName,
						Image:           "non-existing-repo/non-existing-image:v1.0",
						ImagePullPolicy: "Always",
						Command:         []string{"bash"},
						Args:            []string{"-c", `echo "Hello world"`},
					},
				},
			},
		}
		ginkgo.By("creating the pod with invalid image reference and finalizer")
		pod := e2epod.NewPodClient(f).Create(ctx, podSpec)

		ginkgo.By("set up cleanup of the finalizer")
		ginkgo.DeferCleanup(e2epod.NewPodClient(f).RemoveFinalizer, pod.Name, testFinalizer)

		ginkgo.By("Waiting for the pod to be scheduled so that kubelet owns it")
		err := e2epod.WaitForPodScheduled(ctx, f.ClientSet, pod.Namespace, pod.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be scheduled: %q", pod.Name)

		ginkgo.By(fmt.Sprintf("Deleting the pod (%v/%v) to set a deletion timestamp", pod.Namespace, pod.Name))
		err = e2epod.NewPodClient(f).Delete(ctx, pod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "Failed to delete the pod: %q", pod.Name)

		ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be transitioned into the Failed phase", pod.Namespace, pod.Name))
		err = e2epod.WaitForPodTerminatedInNamespace(ctx, f.ClientSet, pod.Name, "", f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be terminated: %q", pod.Name)

		ginkgo.By(fmt.Sprintf("Fetch the end state of the pod (%v/%v)", pod.Namespace, pod.Name))
		pod, err = e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Failed to fetch the end state of the pod: %q", pod.Name)
	})

	ginkgo.DescribeTable("Should transition to Failed phase a deleted pod if non-zero exit codes",
		func(ctx context.Context, policy v1.RestartPolicy) {
			podName := "deleted-running-" + strings.ToLower(string(policy)) + "-" + string(uuid.NewUUID())
			podSpec := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:       podName,
					Finalizers: []string{testFinalizer},
				},
				Spec: v1.PodSpec{
					RestartPolicy: policy,
					Containers: []v1.Container{
						{
							Name:    podName,
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"sleep", "1800"},
						},
					},
				},
			})
			ginkgo.By(fmt.Sprintf("Creating a pod (%v/%v) with restart policy: %v", f.Namespace.Name, podSpec.Name, podSpec.Spec.RestartPolicy))
			pod := e2epod.NewPodClient(f).Create(ctx, podSpec)

			ginkgo.By("set up cleanup of the finalizer")
			ginkgo.DeferCleanup(e2epod.NewPodClient(f).RemoveFinalizer, pod.Name, testFinalizer)

			ginkgo.By("Waiting for the pod to be running")
			err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err, "Failed to await for the pod to be running: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Deleting the pod (%v/%v) to set a deletion timestamp", pod.Namespace, pod.Name))
			err = e2epod.NewPodClient(f).Delete(ctx, pod.Name, *metav1.NewDeleteOptions(1))
			framework.ExpectNoError(err, "Failed to delete the pod: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be transitioned to the failed phase", pod.Namespace, pod.Name))
			err = e2epod.WaitForPodTerminatedInNamespace(ctx, f.ClientSet, pod.Name, "", f.Namespace.Name)
			framework.ExpectNoError(err, "Failed to await for the pod to be terminated: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Fetching the end state of the pod (%v/%v)", pod.Namespace, pod.Name))
			pod, err = e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "Failed to fetch the end state of the pod: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Verify the pod (%v/%v) container is in the terminated state", pod.Namespace, pod.Name))
			gomega.Expect(pod.Status.ContainerStatuses).Should(gomega.HaveLen(1))
			containerStatus := pod.Status.ContainerStatuses[0]
			gomega.Expect(containerStatus.State.Terminated).ToNot(gomega.BeNil(), "The pod container is in not in the Terminated state")

			ginkgo.By(fmt.Sprintf("Verify the pod (%v/%v) container exit code is 137", pod.Namespace, pod.Name))
			gomega.Expect(containerStatus.State.Terminated.ExitCode).Should(gomega.Equal(int32(137)))
		},
		ginkgo.Entry("Restart policy Always", v1.RestartPolicyAlways),
		ginkgo.Entry("Restart policy OnFailure", v1.RestartPolicyOnFailure),
		ginkgo.Entry("Restart policy Never", v1.RestartPolicyNever),
	)

	ginkgo.DescribeTable("Should transition to Succeeded phase a deleted pod when containers complete with 0 exit code",
		func(ctx context.Context, policy v1.RestartPolicy) {
			podName := "deleted-running-" + strings.ToLower(string(policy)) + "-" + string(uuid.NewUUID())
			podSpec := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:       podName,
					Finalizers: []string{testFinalizer},
				},
				Spec: v1.PodSpec{
					RestartPolicy: policy,
					Containers: []v1.Container{
						{
							Name:    podName,
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"sh", "-c"},
							Args: []string{`
							sleep 9999999 &
							PID=$!
							_term() {
								kill $PID
								echo "Caught SIGTERM signal!"
							}

							trap _term SIGTERM
							touch /tmp/trap-marker
							wait $PID

							exit 0
							`,
							},
							ReadinessProbe: &v1.Probe{
								PeriodSeconds: 1,
								ProbeHandler: v1.ProbeHandler{
									Exec: &v1.ExecAction{
										Command: []string{"/bin/sh", "-c", "cat /tmp/trap-marker"},
									},
								},
							},
						},
					},
				},
			})
			ginkgo.By(fmt.Sprintf("Creating a pod (%v/%v) with restart policy: %v", f.Namespace.Name, podSpec.Name, podSpec.Spec.RestartPolicy))
			pod := e2epod.NewPodClient(f).Create(ctx, podSpec)

			ginkgo.By("set up cleanup of the finalizer")
			ginkgo.DeferCleanup(e2epod.NewPodClient(f).RemoveFinalizer, pod.Name, testFinalizer)

			ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be running and with the SIGTERM trap registered", pod.Namespace, pod.Name))
			err := e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name, f.Timeouts.PodStart)
			framework.ExpectNoError(err, "Failed to await for the pod to be running: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Deleting the pod (%v/%v) to set a deletion timestamp", pod.Namespace, pod.Name))
			err = e2epod.NewPodClient(f).Delete(ctx, pod.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Failed to delete the pod: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be transitioned to the succeeded phase", pod.Namespace, pod.Name))
			err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err, "Failed to await for the pod to be succeeded: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Fetching the end state of the pod (%v/%v)", pod.Namespace, pod.Name))
			pod, err = e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "Failed to fetch the end state of the pod: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Verify the pod (%v/%v) container is in the terminated state", pod.Namespace, pod.Name))
			gomega.Expect(pod.Status.ContainerStatuses).Should(gomega.HaveLen(1))
			containerStatus := pod.Status.ContainerStatuses[0]
			gomega.Expect(containerStatus.State.Terminated).ShouldNot(gomega.BeNil(), "The pod container is in not in the Terminated state")

			ginkgo.By(fmt.Sprintf("Verifying the exit code for the terminated container is 0 for pod (%v/%v)", pod.Namespace, pod.Name))
			gomega.Expect(containerStatus.State.Terminated.ExitCode).Should(gomega.Equal(int32(0)))
		},
		ginkgo.Entry("Restart policy Always", v1.RestartPolicyAlways),
		ginkgo.Entry("Restart policy OnFailure", v1.RestartPolicyOnFailure),
		ginkgo.Entry("Restart policy Never", v1.RestartPolicyNever),
	)

	ginkgo.It("Should report true container state when pod have terminationGracePeriodSeconds", func(ctx context.Context) {
		podName := "deletion-grace-period-pod" + string(uuid.NewUUID())
		podSpec := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:       podName,
				Finalizers: []string{testFinalizer},
			},
			Spec: v1.PodSpec{
				RestartPolicy:                 v1.RestartPolicyNever,
				TerminationGracePeriodSeconds: &[]int64{60}[0],
				Containers: []v1.Container{
					{
						Name:    "c1",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sh", "-c"},
						Args: []string{
							`_term() {
            rm -f /tmp/ready
          }
          trap _term SIGTERM
          
          touch /tmp/ready
          
          while true; do
            echo 'helloc1'
            ls /tmp/die_now && echo 'dying in 5s...' && sleep 5 && exit 0
            sleep 1
          done`,
						},
						ReadinessProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								Exec: &v1.ExecAction{
									Command: []string{"/bin/sh", "-c", `if [ -f "/tmp/ready" ]; then
                exit 0
              else
                touch /tmp/die_now
                exit 1
              fi`},
								},
							},
						},
					},
					{
						Name:    "c2",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sh", "-c", `_term() { while true; do echo \"hello_term_c2\"; sleep 1; done } ; trap _term SIGTERM; while true; do echo \"helloc2\"; sleep 1; done`},
					},
				},
			},
		})
		ginkgo.By(fmt.Sprintf("Creating pod %s/%s", f.Namespace.Name, podSpec.Name))
		pod := e2epod.NewPodClient(f).Create(ctx, podSpec)

		ginkgo.By("set up cleanup of the finalizer")
		ginkgo.DeferCleanup(e2epod.NewPodClient(f).RemoveFinalizer, pod.Name, testFinalizer)

		ginkgo.By("waiting for pod to be running")
		err := e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name, f.Timeouts.PodStart)
		framework.ExpectNoError(err, "failed to wait for pod to be ready")

		ginkgo.By(fmt.Sprintf("Deleting the pod (%v/%v) with terminationGracePeriodSeconds to set a deletion timestamp", f.Namespace.Name, pod.Name))
		err = e2epod.NewPodClient(f).Delete(ctx, pod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete the pod: %q", pod.Name)

		// Since TerminationGracePeriodSeconds is set to 60, c1 will be deleted, while c2 will not be deleted immediately.
		// Therefore, before c2 is deleted, the ready status of c1 is false, and the ready status of c2 is true.
		// detail see #129552
		gomega.Eventually(ctx, func() bool {
			pod, _ := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
			c1Ready := pod.Status.ContainerStatuses[0].Ready
			c2Ready := pod.Status.ContainerStatuses[1].Ready

			return c1Ready == false && c2Ready == true
		}, 1*time.Minute, f.Timeouts.Poll).Should(gomega.BeTrueBecause("expect c1 is not ready, c2 is ready"))

		ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be terminated", pod.Namespace, pod.Name))
		err = e2epod.WaitForPodTerminatedInNamespace(ctx, f.ClientSet, pod.Name, "", pod.Namespace)
		framework.ExpectNoError(err, "Failed to await for the pod to be terminated: %q", pod.Name)

		ginkgo.By(fmt.Sprintf("Fetching the end state of the pod (%v/%v)", pod.Namespace, pod.Name))
		pod, err = e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Failed to fetch the end state of the pod: %q", pod.Name)

		ginkgo.By(fmt.Sprintf("Verify the pod (%v/%v) containers are in the terminated state", pod.Namespace, pod.Name))
		gomega.Expect(pod.Status.ContainerStatuses).Should(gomega.HaveLen(2))
		c1Status := pod.Status.ContainerStatuses[0]
		gomega.Expect(c1Status.State.Terminated).ShouldNot(gomega.BeNil(), "The pod container c1 is in the Terminated state")

		c2Status := pod.Status.ContainerStatuses[1]
		gomega.Expect(c2Status.State.Terminated).ShouldNot(gomega.BeNil(), "The pod container c2 is in the Terminated state")
	})

})
