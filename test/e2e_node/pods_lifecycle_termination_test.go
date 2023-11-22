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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

// Pod sigkill test will cover pods with graceful termination period set but failed
// to terminate and forcefully killed by kubelet. This test examine pod's container's
// exit code is 137 and the exit reason is `Error`
var _ = SIGDescribe("Pod SIGKILL [LinuxOnly]", framework.WithNodeConformance(), func() {
	f := framework.NewDefaultFramework("sigkill-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	podName := "sigkill-pod-" + string(uuid.NewUUID())
	containerName := "sigkill-target-container"
	podSpec := getSigkillTargetPod(podName, containerName)
	ginkgo.Context("", func() {
		ginkgo.BeforeEach(func() {
			ginkgo.By("setting up the pod to be used in the test")
			e2epod.NewPodClient(f).Create(context.TODO(), podSpec)
		})

		ginkgo.It("The containers terminated forcefully by Sigkill should have the correct exit code(137) and reason (Error)", func() {

			ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be running", f.Namespace.Name, podSpec.Name))
			err := e2epod.WaitForPodNameRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, f.Namespace.Name)
			framework.ExpectNoError(err, "Failed to await for the pod to be running: %q", podSpec.Name)

			// Checking pod's readiness to confirm the signal handler has registered successfully.
			err = e2epod.WaitForPodCondition(context.TODO(), f.ClientSet, f.Namespace.Name, podSpec.Name, "Ready", f.Timeouts.PodStart, testutils.PodRunningReady)
			framework.ExpectNoError(err, "Failed to await Pod (%v/%v) become ready after registering signal handler: %v", f.Namespace.Name, podSpec.Name, err)

			ginkgo.By(fmt.Sprintf("Deleting the pod (%v/%v) to set a deletion timestamp", f.Namespace.Name, podSpec.Name))
			err = e2epod.NewPodClient(f).Delete(context.TODO(), podSpec.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Failed to delete the pod: %q", podSpec.Name)

			ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be transitioned to the terminated phase", f.Namespace.Name, podSpec.Name))
			err = e2epod.WaitForPodTerminatedInNamespace(context.TODO(), f.ClientSet, podSpec.Name, "", f.Namespace.Name)
			framework.ExpectNoError(err, "Failed to await for the pod to be terminated: %q", podSpec.Name)

			ginkgo.By(fmt.Sprintf("Fetching the end state of the pod (%v/%v)", f.Namespace.Name, podSpec.Name))
			pod, err := e2epod.NewPodClient(f).Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "Failed to fetch the end state of the pod: %q", podSpec.Name)

			ginkgo.By(fmt.Sprintf("Verify the pod (%v/%v) container is in the terminated state", pod.Namespace, podSpec.Name))
			gomega.Expect(pod.Status.ContainerStatuses).Should(gomega.HaveLen(1), "The pod container has %v status", len(pod.Status.ContainerStatuses))
			containerStatus := pod.Status.ContainerStatuses[0]
			gomega.Expect(containerStatus.State.Terminated).ShouldNot(gomega.BeNil(), "The pod container is in not in the Terminated state")

			ginkgo.By(fmt.Sprintf("Verifying the exit code for the terminated container is 137 for pod (%v/%v)", pod.Namespace, podSpec.Name))
			gomega.Expect(containerStatus.State.Terminated.ExitCode).Should(gomega.Equal(int32(137)))

			ginkgo.By(fmt.Sprintf("Verify exit reason of the pod (%v/%v) container", f.Namespace.Name, podSpec.Name))
			gomega.Expect(containerStatus.State.Terminated.Reason).Should(gomega.Equal("Error"), "Container terminated by sigkill expect Error but got %v", containerStatus.State.Terminated.Reason)
		})

		ginkgo.AfterEach(func() {
			ginkgo.By(fmt.Sprintf("Deleting pod by removing finalizers: %s", podSpec.Name))
			e2epod.NewPodClient(f).RemoveFinalizer(context.TODO(), podSpec.Name, testFinalizer)

			ginkgo.By(fmt.Sprintf("Confirm the pod was successfully deleted: %s", podSpec.Name))
			e2epod.WaitForPodNotFoundInNamespace(context.TODO(), f.ClientSet, podSpec.Name, f.Namespace.Name, f.Timeouts.PodDelete)
		})
	})
})

func getSigkillTargetPod(podName string, ctnName string) *v1.Pod {
	gracePeriod := int64(5)
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
			// Using default test finalizer to keep exit status and code can be
			// preserved after deleting the pod
			Finalizers: []string{testFinalizer},
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  ctnName,
					Image: busyboxImage,
					// In the main container, SIGTERM was trapped and later /tmp/healthy
					// will be created for readiness probe to verify if the trap was
					// executed successfully
					Command: []string{
						"sh",
						"-c",
						"trap \"echo SIGTERM caught\" SIGTERM SIGINT; touch /tmp/healthy; /bin/sleep 1000",
					},
					// Using readiness probe to guarantee signal handler registering finished
					ReadinessProbe: &v1.Probe{
						InitialDelaySeconds: 1,
						TimeoutSeconds:      2,
						ProbeHandler: v1.ProbeHandler{
							Exec: &v1.ExecAction{
								Command: []string{"/bin/sh", "-c", "cat /tmp/healthy"},
							},
						},
					},
				},
			},
			TerminationGracePeriodSeconds: &gracePeriod,
		},
	}
}
