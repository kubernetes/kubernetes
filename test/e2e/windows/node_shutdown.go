/*
Copyright 2024 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubectl/pkg/util/podutils"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"

	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = sigDescribe(feature.Windows, "GracefulNodeShutdown", framework.WithSerial(), framework.WithDisruptive(), framework.WithSlow(), skipUnlessWindows(func() {
	f := framework.NewDefaultFramework("windows-node-graceful-shutdown")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should be able to gracefully shutdown pods with various grace periods", func(ctx context.Context) {
		const (
			pollInterval                        = 1 * time.Second
			podStatusUpdateTimeout              = 90 * time.Second
			nodeStatusUpdateTimeout             = 90 * time.Second
			nodeShutdownGracePeriod             = 20 * time.Second
			nodeShutdownGracePeriodCriticalPods = 10 * time.Second
		)

		ginkgo.By("selecting a Windows node")
		targetNode, err := findWindowsNode(ctx, f)
		framework.ExpectNoError(err, "Error finding Windows node")
		framework.Logf("Using node: %v", targetNode.Name)

		nodeName := targetNode.Name
		nodeSelector := fields.Set{
			"spec.nodeName": nodeName,
		}.AsSelector().String()

		// Define test pods
		pods := []*v1.Pod{
			getGracePeriodOverrideTestPod("period-20-"+string(uuid.NewUUID()), nodeName, 20, ""),
			getGracePeriodOverrideTestPod("period-25-"+string(uuid.NewUUID()), nodeName, 25, ""),
			getGracePeriodOverrideTestPod("period-critical-5-"+string(uuid.NewUUID()), nodeName, 5, scheduling.SystemNodeCritical),
			getGracePeriodOverrideTestPod("period-critical-10-"+string(uuid.NewUUID()), nodeName, 10, scheduling.SystemNodeCritical),
		}

		ginkgo.By("Creating batch pods")
		e2epod.NewPodClient(f).CreateBatch(ctx, pods)

		list, err := e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{
			FieldSelector: nodeSelector,
		})
		framework.ExpectNoError(err)
		gomega.Expect(list.Items).To(gomega.HaveLen(len(pods)), "the number of pods is not as expected")

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		go func() {
			defer ginkgo.GinkgoRecover()
			w := &cache.ListWatch{
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					return f.ClientSet.CoreV1().Pods(f.Namespace.Name).Watch(ctx, options)
				},
			}

			// Setup watch to continuously monitor any pod events and detect invalid pod status updates
			_, err = watchtools.Until(ctx, list.ResourceVersion, w, func(event watch.Event) (bool, error) {
				if pod, ok := event.Object.(*v1.Pod); ok {
					if isPodReadyWithFailedStatus(pod) {
						return false, fmt.Errorf("failing test due to detecting invalid pod status")
					}
					// Watch will never terminate (only when the test ends due to context cancellation)
					return false, nil
				}
				return false, nil
			})

			// Ignore timeout error since the context will be explicitly cancelled and the watch will never return true
			if err != nil && !wait.Interrupted(err) {
				framework.Failf("watch for invalid pod status failed: %v", err.Error())
			}
		}()

		ginkgo.By("Verifying batch pods are running")
		for _, pod := range list.Items {
			if podReady, err := testutils.PodRunningReady(&pod); err != nil || !podReady {
				framework.Failf("Failed to start batch pod: %v", pod.Name)
			}
		}

		for _, pod := range list.Items {
			framework.Logf("Pod (%v/%v) status conditions: %q", pod.Namespace, pod.Name, &pod.Status.Conditions)
		}

		// use to keep the node active before testing critical pods reaching the terminate state
		delyapodName := "delay-shutdown-20-" + string(uuid.NewUUID())
		delayPod := getGracePeriodOverrideTestPod(delyapodName, nodeName, 20, scheduling.SystemNodeCritical)
		e2epod.NewPodClient(f).CreateSync(ctx, delayPod)

		ginkgo.By("Emitting shutdown signal")

		emitSignalPrepareForShutdown(nodeName, f, ctx)

		ginkgo.By("Verifying that non-critical pods are shutdown")
		// Non critical pod should be shutdown
		gomega.Eventually(ctx, func(ctx context.Context) error {
			list, err = e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{
				FieldSelector: nodeSelector,
			})
			if err != nil {
				return err
			}
			// Find pods with names starting with "period"
			filteredPods := []v1.Pod{}
			for _, pod := range list.Items {
				if strings.HasPrefix(pod.Name, "period") {
					filteredPods = append(filteredPods, pod)
				}
			}
			gomega.Expect(filteredPods).To(gomega.HaveLen(len(pods)), "the number of pods is not as expected")

			for _, pod := range filteredPods {
				if kubelettypes.IsCriticalPod(&pod) {
					if isPodShutdown(&pod) {
						framework.Logf("Expecting critical pod (%v/%v) to be running, but it's not currently. Pod Status %+v", pod.Namespace, pod.Name, pod.Status)
						return fmt.Errorf("critical pod (%v/%v) should not be shutdown, phase: %s", pod.Namespace, pod.Name, pod.Status.Phase)
					}
				} else {
					if !isPodShutdown(&pod) {
						framework.Logf("Expecting non-critical pod (%v/%v) to be shutdown, but it's not currently. Pod Status %+v", pod.Namespace, pod.Name, pod.Status)
						return fmt.Errorf("pod (%v/%v) should be shutdown, phase: %s", pod.Namespace, pod.Name, pod.Status.Phase)
					}
				}
			}
			return nil
		}, podStatusUpdateTimeout, pollInterval).Should(gomega.Succeed())

		ginkgo.By("Verifying that all pods are shutdown")
		// All pod should be shutdown
		gomega.Eventually(ctx, func(ctx context.Context) error {
			list, err = e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{
				FieldSelector: nodeSelector,
			})
			if err != nil {
				return err
			}
			// Find pods with names starting with "period"
			filteredPods := []v1.Pod{}
			for _, pod := range list.Items {
				if strings.HasPrefix(pod.Name, "period") {
					filteredPods = append(filteredPods, pod)
				}
			}

			gomega.Expect(filteredPods).To(gomega.HaveLen(len(pods)), "the number of pods is not as expected")

			for _, pod := range filteredPods {
				if !isPodShutdown(&pod) {
					framework.Logf("Expecting pod (%v/%v) to be shutdown, but it's not currently: Pod Status %+v", pod.Namespace, pod.Name, pod.Status)
					return fmt.Errorf("pod (%v/%v) should be shutdown, phase: %s", pod.Namespace, pod.Name, pod.Status.Phase)
				}
			}
			return nil
		},
			// Critical pod starts shutdown after (nodeShutdownGracePeriod-nodeShutdownGracePeriodCriticalPods)
			podStatusUpdateTimeout+(nodeShutdownGracePeriod-nodeShutdownGracePeriodCriticalPods),
			pollInterval).Should(gomega.Succeed())

		ginkgo.By("Verify that all pod ready to start condition are set to false after terminating")
		// All pod ready to start condition should set to false
		gomega.Eventually(ctx, func(ctx context.Context) error {
			list, err = e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{
				FieldSelector: nodeSelector,
			})
			if err != nil {
				return err
			}

			// Find pods with names starting with "period"
			filteredPods := []v1.Pod{}
			for _, pod := range list.Items {
				if strings.HasPrefix(pod.Name, "period") {
					filteredPods = append(filteredPods, pod)
				}
			}
			gomega.Expect(filteredPods).To(gomega.HaveLen(len(pods)), "the number of pods is not as expected")

			for _, pod := range filteredPods {
				if !isPodReadyToStartConditionSetToFalse(&pod) {
					framework.Logf("Expecting pod (%v/%v) 's ready to start condition set to false, "+
						"but it's not currently: Pod Condition %+v", pod.Namespace, pod.Name, pod.Status.Conditions)
					return fmt.Errorf("pod (%v/%v) 's ready to start condition should be false, condition: %v, phase: %s",
						pod.Namespace, pod.Name, pod.Status.Conditions, pod.Status.Phase)
				}
			}
			return nil
		},
		).Should(gomega.Succeed())
	})
}))

// getGracePeriodOverrideTestPod returns a new Pod object containing a container
// runs a shell script, hangs the process until a SIGTERM signal is received.
// The script waits for $PID to ensure that the process does not exist.
// If priorityClassName is scheduling.SystemNodeCritical, the Pod is marked as critical and a comment is added.
func getGracePeriodOverrideTestPod(name string, node string, gracePeriod int64, priorityClassName string) *v1.Pod {
	agnhostImage := imageutils.GetE2EImage(imageutils.Agnhost)

	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    name,
					Image:   agnhostImage,
					Command: []string{"/agnhost", "netexec", "--delay-shutdown", "9999"},
				},
			},
			TerminationGracePeriodSeconds: &gracePeriod,
			NodeName:                      node,
		},
	}
	if priorityClassName == scheduling.SystemNodeCritical {
		pod.ObjectMeta.Annotations = map[string]string{
			kubelettypes.ConfigSourceAnnotationKey: kubelettypes.FileSource,
		}
		pod.Spec.PriorityClassName = priorityClassName
		if !kubelettypes.IsCriticalPod(pod) {
			framework.Failf("pod %q should be a critical pod", pod.Name)
		}
	} else {
		pod.Spec.PriorityClassName = priorityClassName
		if kubelettypes.IsCriticalPod(pod) {
			framework.Failf("pod %q should not be a critical pod", pod.Name)
		}
	}
	return pod
}

// Emits a reboot event from HPC. Will cause kubelet to react to an active shutdown event.
func emitSignalPrepareForShutdown(nodeName string, f *framework.Framework, ctx context.Context) {
	ginkgo.By("scheduling a pod with a container that emits a PrepareForShutdown signal")

	windowsImage := imageutils.GetE2EImage(imageutils.Agnhost)

	trueVar := true
	podName := "reboot-host-test-pod"
	user := "NT AUTHORITY\\SYSTEM"
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{
				WindowsOptions: &v1.WindowsSecurityContextOptions{
					HostProcess:   &trueVar,
					RunAsUserName: &user,
				},
			},
			HostNetwork: true,
			Containers: []v1.Container{
				{
					Image: windowsImage,
					Name:  "reboot-computer-test",
					Command: []string{
						"powershell.exe",
						"-Command",
						"$os = Get-WmiObject -Class win32_operatingsystem;",
						"[Environment]::SetEnvironmentVariable(\"TMP_BOOT_DATE\", $os.LastBootUpTime, \"Machine\");",
						"[Environment]::SetEnvironmentVariable(\"TMP_INSTALL_DATE\", $os.InstallDate, \"Machine\");",
						"shutdown.exe -r -t 60",
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			NodeName:      nodeName,
		},
	}

	e2epod.NewPodClient(f).Create(ctx, pod)

	ginkgo.By("Waiting for pod to run")
	e2epod.NewPodClient(f).WaitForFinish(ctx, podName, 3*time.Minute)

	ginkgo.By("Then ensuring pod finished running successfully")
	p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(
		ctx,
		podName,
		metav1.GetOptions{})

	framework.ExpectNoError(err, "Error retrieving pod")
	gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodSucceeded))
}

const (
	// https://github.com/kubernetes/kubernetes/blob/1dd781ddcad454cc381806fbc6bd5eba8fa368d7/pkg/kubelet/nodeshutdown/nodeshutdown_manager_linux.go#L43-L44
	podShutdownReason  = "Terminated"
	podShutdownMessage = "Pod was terminated in response to imminent node shutdown."
)

func isPodShutdown(pod *v1.Pod) bool {
	if pod == nil {
		return false
	}

	hasContainersNotReadyCondition := false
	for _, cond := range pod.Status.Conditions {
		if cond.Type == v1.ContainersReady && cond.Status == v1.ConditionFalse {
			hasContainersNotReadyCondition = true
		}
	}

	return pod.Status.Message == podShutdownMessage && pod.Status.Reason == podShutdownReason && hasContainersNotReadyCondition && pod.Status.Phase == v1.PodFailed
}

// Pods should never report failed phase and have ready condition = true (https://github.com/kubernetes/kubernetes/issues/108594)
func isPodReadyWithFailedStatus(pod *v1.Pod) bool {
	return pod.Status.Phase == v1.PodFailed && podutils.IsPodReady(pod)
}

func isPodReadyToStartConditionSetToFalse(pod *v1.Pod) bool {
	if pod == nil {
		return false
	}
	readyToStartConditionSetToFalse := false
	for _, cond := range pod.Status.Conditions {
		if cond.Status == v1.ConditionFalse {
			readyToStartConditionSetToFalse = true
		}
	}

	return readyToStartConditionSetToFalse
}
