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

package windows

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	testutils "k8s.io/kubernetes/test/utils"
	"k8s.io/kubernetes/test/utils/format"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

const runAsUserNameContainerName = "run-as-username-container"

var _ = sigDescribe(feature.Windows, "SecurityContext", skipUnlessWindows(func() {
	f := framework.NewDefaultFramework("windows-run-as-username")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should be able create pods and run containers with a given username", func(ctx context.Context) {
		ginkgo.By("Creating 2 pods: 1 with the default user, and one with a custom one.")
		podDefault := runAsUserNamePod(nil)
		e2eoutput.TestContainerOutput(ctx, f, "check default user", podDefault, 0, []string{"ContainerUser"})

		podUserName := runAsUserNamePod(toPtr("ContainerAdministrator"))
		e2eoutput.TestContainerOutput(ctx, f, "check set user", podUserName, 0, []string{"ContainerAdministrator"})
	})

	ginkgo.It("should not be able to create pods with unknown usernames at Pod level", func(ctx context.Context) {
		ginkgo.By("Creating a pod with an invalid username")
		podInvalid := e2epod.NewPodClient(f).Create(ctx, runAsUserNamePod(toPtr("FooLish")))

		failedSandboxEventSelector := fields.Set{
			"involvedObject.kind":      "Pod",
			"involvedObject.name":      podInvalid.Name,
			"involvedObject.namespace": podInvalid.Namespace,
			"reason":                   events.FailedCreatePodSandBox,
		}.AsSelector().String()
		hcsschimError := "The user name or password is incorrect."

		// Hostprocess updated the cri to pass RunAsUserName to sandbox: https://github.com/kubernetes/kubernetes/pull/99576/commits/51a02fdb80cb7ba042a66362eb76facd2fd82401
		// Some runtimes might use that and set the username on the podsandbox. Containerd 1.6+ is known to do this.
		// If there is an error when creating the pod sandbox then the pod stays in pending state by design
		// See https://github.com/kubernetes/kubernetes/issues/104635
		// Not all runtimes use the sandbox information.  This means the test needs to check if the pod
		// sandbox failed or workload pod failed.
		framework.Logf("Waiting for pod %s to enter the error state.", podInvalid.Name)
		gomega.Eventually(ctx, func(ctx context.Context) bool {
			failedSandbox, err := eventOccurred(ctx, f.ClientSet, podInvalid.Namespace, failedSandboxEventSelector, hcsschimError)
			if err != nil {
				framework.Logf("Error retrieving events for pod. Ignoring...")
			}
			if failedSandbox {
				framework.Logf("Found Expected Event 'Failed to Create Pod Sandbox' with message containing: %s", hcsschimError)
				return true
			}

			framework.Logf("No Sandbox error found. Looking for failure in workload pods")
			pod, err := e2epod.NewPodClient(f).Get(ctx, podInvalid.Name, metav1.GetOptions{})
			if err != nil {
				framework.Logf("Error retrieving pod: %s", err)
				return false
			}

			podTerminatedReason := testutils.TerminatedContainers(pod)[runAsUserNameContainerName]
			podFailedToStart := podTerminatedReason == "ContainerCannotRun" || podTerminatedReason == "StartError"
			if pod.Status.Phase == v1.PodFailed && podFailedToStart {
				framework.Logf("Found terminated workload Pod that could not start")
				return true
			}

			return false
		}, framework.PodStartTimeout, 1*time.Second).Should(gomega.BeTrue())
	})

	ginkgo.It("should not be able to create pods with unknown usernames at Container level", func(ctx context.Context) {
		ginkgo.By("Creating a pod with an invalid username at container level and pod running as ContainerUser")
		p := runAsUserNamePod(toPtr("FooLish"))
		p.Spec.SecurityContext.WindowsOptions.RunAsUserName = toPtr("ContainerUser")
		podInvalid := e2epod.NewPodClient(f).Create(ctx, p)

		framework.Logf("Waiting for pod %s to enter the error state.", podInvalid.Name)
		framework.ExpectNoError(e2epod.WaitForPodTerminatedInNamespace(ctx, f.ClientSet, podInvalid.Name, "", f.Namespace.Name))

		podInvalid, _ = e2epod.NewPodClient(f).Get(ctx, podInvalid.Name, metav1.GetOptions{})
		podTerminatedReason := testutils.TerminatedContainers(podInvalid)[runAsUserNameContainerName]
		if podTerminatedReason != "ContainerCannotRun" && podTerminatedReason != "StartError" {
			framework.Failf("The container terminated reason was supposed to be: 'ContainerCannotRun' or 'StartError', not: '%q'", podTerminatedReason)
		}
	})

	ginkgo.It("should override SecurityContext username if set", func(ctx context.Context) {
		ginkgo.By("Creating a pod with 2 containers with different username configurations.")

		pod := runAsUserNamePod(toPtr("ContainerAdministrator"))
		pod.Spec.Containers[0].SecurityContext.WindowsOptions.RunAsUserName = toPtr("ContainerUser")
		pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
			Name:    "run-as-username-new-container",
			Image:   imageutils.GetE2EImage(imageutils.NonRoot),
			Command: []string{"cmd", "/S", "/C", "echo %username%"},
		})

		e2eoutput.TestContainerOutput(ctx, f, "check overridden username", pod, 0, []string{"ContainerUser"})
		e2eoutput.TestContainerOutput(ctx, f, "check pod SecurityContext username", pod, 1, []string{"ContainerAdministrator"})
	})

	ginkgo.It("should ignore Linux Specific SecurityContext if set", func(ctx context.Context) {
		ginkgo.By("Creating a pod with SELinux options")
		// It is sufficient to show that the pod comes up here. Since we're stripping the SELinux and other linux
		// security contexts in apiserver and not updating the pod object in the apiserver, we cannot validate the
		// pod object to not have those security contexts. However the pod coming to running state is a sufficient
		// enough condition for us to validate since prior to https://github.com/kubernetes/kubernetes/pull/93475
		// the pod would have failed to come up.
		windowsPodWithSELinux := createTestPod(f, imageutils.GetE2EImage(imageutils.Agnhost), windowsOS)
		windowsPodWithSELinux.Spec.Containers[0].Args = []string{"test-webserver-with-selinux"}
		windowsPodWithSELinux.Spec.SecurityContext = &v1.PodSecurityContext{}
		containerUserName := "ContainerAdministrator"
		windowsPodWithSELinux.Spec.SecurityContext.SELinuxOptions = &v1.SELinuxOptions{Level: "s0:c24,c9"}
		windowsPodWithSELinux.Spec.Containers[0].SecurityContext = &v1.SecurityContext{
			SELinuxOptions: &v1.SELinuxOptions{Level: "s0:c24,c9"},
			WindowsOptions: &v1.WindowsSecurityContextOptions{RunAsUserName: &containerUserName}}
		windowsPodWithSELinux.Spec.Tolerations = []v1.Toleration{{Key: "os", Value: "Windows"}}
		windowsPodWithSELinux, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx,
			windowsPodWithSELinux, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.Logf("Created pod %v", windowsPodWithSELinux)
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, windowsPodWithSELinux.Name,
			f.Namespace.Name), "failed to wait for pod %s to be running", windowsPodWithSELinux.Name)
	})

	ginkgo.It("should not be able to create pods with containers running as ContainerAdministrator when runAsNonRoot is true", func(ctx context.Context) {
		ginkgo.By("Creating a pod")

		p := runAsUserNamePod(toPtr("ContainerAdministrator"))
		p.Spec.SecurityContext.RunAsNonRoot = &trueVar

		podInvalid, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, p, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Error creating pod")

		ginkgo.By("Waiting for pod to finish")
		event, err := e2epod.NewPodClient(f).WaitForErrorEventOrSuccess(ctx, podInvalid)
		framework.ExpectNoError(err)
		gomega.Expect(event).ToNot(gomega.BeNil(), "event should not be empty")
		framework.Logf("Got event: %v", event)
		expectedEventError := "container's runAsUserName (ContainerAdministrator) which will be regarded as root identity and will break non-root policy"
		gomega.Expect(event.Message).Should(gomega.ContainSubstring(expectedEventError), "Event error should indicate non-root policy caused container to not start")
	})

	ginkgo.It("should not be able to create pods with containers running as CONTAINERADMINISTRATOR when runAsNonRoot is true", func(ctx context.Context) {
		ginkgo.By("Creating a pod")

		p := runAsUserNamePod(toPtr("CONTAINERADMINISTRATOR"))
		p.Spec.SecurityContext.RunAsNonRoot = &trueVar

		podInvalid, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, p, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Error creating pod")

		ginkgo.By("Waiting for pod to finish")
		event, err := e2epod.NewPodClient(f).WaitForErrorEventOrSuccess(ctx, podInvalid)
		framework.ExpectNoError(err)
		gomega.Expect(event).ToNot(gomega.BeNil(), "event should not be empty")
		framework.Logf("Got event: %v", event)
		expectedEventError := "container's runAsUserName (CONTAINERADMINISTRATOR) which will be regarded as root identity and will break non-root policy"
		gomega.Expect(event.Message).Should(gomega.ContainSubstring(expectedEventError), "Event error should indicate non-root policy caused container to not start")
	})
}))

var _ = sigDescribe(feature.Windows, "SecurityContext", skipUnlessWindows(func() {
	f := framework.NewDefaultFramework("windows-with-unsupported-fields")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should be able to create pod and run containers", func(ctx context.Context) {
		ginkgo.By("Creating 1 pods: run with unsupported fields")

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "run-ignore-unsupported-fields",
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				NodeSelector: map[string]string{"kubernetes.io/os": "windows"},
				Containers: []v1.Container{
					{
						Name:  "test-container",
						Image: imageutils.GetE2EImage(imageutils.Pause),
					},
				},
				SecurityContext: &v1.PodSecurityContext{
					RunAsUser:    ptr.To[int64](999), // windows does not support
					RunAsGroup:   ptr.To[int64](999), // windows does not support
					RunAsNonRoot: ptr.To(true),
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Error creating pod")

		podErr := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)

		// Get the logs and events before calling ExpectNoError, so we can debug any errors.
		var logs string
		var events *v1.EventList
		if err := wait.PollUntilContextTimeout(ctx, 30*time.Second, 2*time.Minute, true, func(ctx context.Context) (done bool, err error) {
			framework.Logf("polling logs")
			logs, err = e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
			if err != nil {
				framework.Logf("Error pulling logs: %v", err)
				return false, nil
			}

			events, err = f.ClientSet.CoreV1().Events(pod.Namespace).Search(scheme.Scheme, pod)
			if err != nil {
				return false, fmt.Errorf("error in listing events: %w", err)
			}
			return true, nil
		}); err != nil {
			framework.Failf("Unexpected error getting pod logs/events: %v", err)
		} else {
			framework.Logf("Pod logs: \n%v", logs)
			framework.Logf("Pod events: \n%v", format.Object(events, 1))
		}

		framework.ExpectNoError(podErr)
	})
}))

func runAsUserNamePod(username *string) *v1.Pod {
	podName := "run-as-username-" + string(uuid.NewUUID())
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			NodeSelector: map[string]string{"kubernetes.io/os": "windows"},
			Containers: []v1.Container{
				{
					Name:    runAsUserNameContainerName,
					Image:   imageutils.GetE2EImage(imageutils.NonRoot),
					Command: []string{"cmd", "/S", "/C", "echo %username%"},
					SecurityContext: &v1.SecurityContext{
						WindowsOptions: &v1.WindowsSecurityContextOptions{
							RunAsUserName: username,
						},
					},
				},
			},
			SecurityContext: &v1.PodSecurityContext{
				WindowsOptions: &v1.WindowsSecurityContextOptions{
					RunAsUserName: username,
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
}

func toPtr(s string) *string {
	return &s
}

func eventOccurred(ctx context.Context, c clientset.Interface, namespace, eventSelector, msg string) (bool, error) {
	options := metav1.ListOptions{FieldSelector: eventSelector}

	events, err := c.CoreV1().Events(namespace).List(ctx, options)
	if err != nil {
		return false, fmt.Errorf("got error while getting events: %w", err)
	}
	for _, event := range events.Items {
		if strings.Contains(event.Message, msg) {
			return true, nil
		}
	}
	return false, nil
}
