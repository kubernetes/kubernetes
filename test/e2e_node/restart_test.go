//go:build linux

/*
Copyright 2015 The Kubernetes Authors.

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
	"os/exec"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/util/uuid"
)

type podCondition func(pod *v1.Pod) (bool, error)

// waitForPodsCondition waits for `podCount` number of pods to match a specific pod condition within a timeout duration.
// If the timeout is hit, it returns the list of currently running pods.
func waitForPodsCondition(ctx context.Context, f *framework.Framework, podCount int, timeout time.Duration, condition podCondition) (runningPods []*v1.Pod) {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(10 * time.Second) {
		podList, err := e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{})
		if err != nil {
			framework.Logf("Failed to list pods on node: %v", err)
			continue
		}

		runningPods = []*v1.Pod{}
		for i := range podList.Items {
			pod := podList.Items[i]
			if r, err := condition(&pod); err != nil || !r {
				continue
			}
			runningPods = append(runningPods, &pod)
		}
		framework.Logf("Running pod count %d", len(runningPods))
		if len(runningPods) >= podCount {
			break
		}
	}
	return runningPods
}

var _ = SIGDescribe("Restart", framework.WithSerial(), framework.WithSlow(), framework.WithDisruptive(), func() {
	const (
		// Saturate the node. It's not necessary that all these pods enter
		// Running/Ready, because we don't know the number of cores in the
		// test node or default limits applied (if any). It's is essential
		// that no containers end up in terminated. 100 was chosen because
		// it's the max pods per node.
		podCount            = 100
		podCreationInterval = 100 * time.Millisecond
		recoverTimeout      = 5 * time.Minute
		startTimeout        = 3 * time.Minute
		// restartCount is chosen so even with minPods we exhaust the default
		// allocation of a /24.
		minPods      = 50
		restartCount = 6
	)

	f := framework.NewDefaultFramework("restart-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.Context("Container Runtime", func() {
		ginkgo.Context("Network", func() {
			ginkgo.It("should recover from ip leak", func(ctx context.Context) {
				pods := newTestPods(podCount, false, imageutils.GetPauseImageName(), "restart-container-runtime-test")
				ginkgo.By(fmt.Sprintf("Trying to create %d pods on node", len(pods)))
				createBatchPodWithRateControl(ctx, f, pods, podCreationInterval)
				ginkgo.DeferCleanup(deletePodsSync, f, pods)

				// Give the node some time to stabilize, assume pods that enter RunningReady within
				// startTimeout fit on the node and the node is now saturated.
				runningPods := waitForPodsCondition(ctx, f, podCount, startTimeout, testutils.PodRunningReadyOrSucceeded)
				if len(runningPods) < minPods {
					framework.Failf("Failed to start %d pods, cannot test that restarting container runtime doesn't leak IPs", minPods)
				}

				for i := 0; i < restartCount; i++ {
					ginkgo.By(fmt.Sprintf("Killing container runtime iteration %d", i))
					// Wait for container runtime to be running
					var pid int
					gomega.Eventually(ctx, func() error {
						runtimePids, err := getPidsForProcess(framework.TestContext.ContainerRuntimeProcessName, framework.TestContext.ContainerRuntimePidFile)
						if err != nil {
							return err
						}
						if len(runtimePids) != 1 {
							return fmt.Errorf("unexpected container runtime pid list: %+v", runtimePids)
						}
						// Make sure the container runtime is running, pid got from pid file may not be running.
						pid = runtimePids[0]
						if _, err := exec.Command("sudo", "ps", "-p", fmt.Sprintf("%d", pid)).CombinedOutput(); err != nil {
							return err
						}
						return nil
					}, 1*time.Minute, 2*time.Second).Should(gomega.BeNil())
					if stdout, err := exec.Command("sudo", "kill", "-SIGKILL", fmt.Sprintf("%d", pid)).CombinedOutput(); err != nil {
						framework.Failf("Failed to kill container runtime (pid=%d): %v, stdout: %q", pid, err, string(stdout))
					}
					// Assume that container runtime will be restarted by systemd/supervisord etc.
					time.Sleep(20 * time.Second)
				}

				ginkgo.By("Checking currently Running/Ready pods")
				postRestartRunningPods := waitForPodsCondition(ctx, f, len(runningPods), recoverTimeout, testutils.PodRunningReadyOrSucceeded)
				if len(postRestartRunningPods) == 0 {
					framework.Failf("Failed to start *any* pods after container runtime restart, this might indicate an IP leak")
				}
				ginkgo.By("Confirm no containers have terminated")
				for _, pod := range postRestartRunningPods {
					if c := testutils.TerminatedContainers(pod); len(c) != 0 {
						framework.Failf("Pod %q has failed containers %+v after container runtime restart, this might indicate an IP leak", pod.Name, c)
					}
				}
				ginkgo.By(fmt.Sprintf("Container runtime restart test passed with %d pods", len(postRestartRunningPods)))
			})
		})
	})
	ginkgo.Context("Kubelet", func() {
		ginkgo.It("should correctly account for terminated pods after restart", func(ctx context.Context) {
			node := getLocalNode(ctx, f)
			cpus := node.Status.Allocatable[v1.ResourceCPU]
			numCpus := int((&cpus).Value())
			if numCpus < 1 {
				e2eskipper.Skipf("insufficient CPU available for kubelet restart test")
			}
			if numCpus > 18 {
				// 950m * 19 = 1805 CPUs -> not enough to block the scheduling of another 950m pod
				e2eskipper.Skipf("test will return false positives on a machine with >18 cores")
			}

			// create as many restartNever pods as there are allocatable CPU
			// nodes; if they are not correctly accounted for as terminated
			// later, this will fill up all node capacity
			podCountRestartNever := numCpus
			ginkgo.By(fmt.Sprintf("creating %d RestartNever pods on node", podCountRestartNever))
			restartNeverPods := newTestPods(podCountRestartNever, false, imageutils.GetE2EImage(imageutils.BusyBox), "restart-kubelet-test")
			for _, pod := range restartNeverPods {
				pod.Spec.RestartPolicy = "Never"
				pod.Spec.Containers[0].Command = []string{"echo", "hi"}
				pod.Spec.Containers[0].Resources.Limits = v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("950m"), // leave a little room for other workloads
				}
			}
			createBatchPodWithRateControl(ctx, f, restartNeverPods, podCreationInterval)
			ginkgo.DeferCleanup(deletePodsSync, f, restartNeverPods)
			completedPods := waitForPodsCondition(ctx, f, podCountRestartNever, startTimeout, testutils.PodSucceeded)

			if len(completedPods) < podCountRestartNever {
				framework.Failf("Failed to run sufficient restartNever pods, got %d but expected %d", len(completedPods), podCountRestartNever)
			}

			podCountRestartAlways := (numCpus / 2) + 1
			ginkgo.By(fmt.Sprintf("creating %d RestartAlways pods on node", podCountRestartAlways))
			restartAlwaysPods := newTestPods(podCountRestartAlways, false, imageutils.GetPauseImageName(), "restart-kubelet-test")
			for _, pod := range restartAlwaysPods {
				pod.Spec.Containers[0].Resources.Limits = v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("1"),
				}
			}
			createBatchPodWithRateControl(ctx, f, restartAlwaysPods, podCreationInterval)
			ginkgo.DeferCleanup(deletePodsSync, f, restartAlwaysPods)

			numAllPods := podCountRestartNever + podCountRestartAlways
			allPods := waitForPodsCondition(ctx, f, numAllPods, startTimeout, testutils.PodRunningReadyOrSucceeded)
			if len(allPods) < numAllPods {
				framework.Failf("Failed to run sufficient restartAlways pods, got %d but expected %d", len(allPods), numAllPods)
			}

			ginkgo.By("killing and restarting kubelet")
			// We want to kill the kubelet rather than a graceful restart
			restartKubelet := mustStopKubelet(ctx, f)
			restartKubelet(ctx)

			// If this test works correctly, each of these pods will exit
			// with no issue. But if accounting breaks, pods scheduled after
			// restart may think these old pods are consuming CPU and we
			// will get an OutOfCpu error.
			ginkgo.By("verifying restartNever pods succeed and restartAlways pods stay running")
			for start := time.Now(); time.Since(start) < startTimeout && ctx.Err() == nil; time.Sleep(10 * time.Second) {
				postRestartRunningPods := waitForPodsCondition(ctx, f, numAllPods, recoverTimeout, testutils.PodRunningReadyOrSucceeded)
				if len(postRestartRunningPods) < numAllPods {
					framework.Failf("less pods are running after node restart, got %d but expected %d", len(postRestartRunningPods), numAllPods)
				}
			}
		})
		// Regression test for https://issues.k8s.io/116925
		ginkgo.It("should delete pods which are marked as terminal and have a deletion timestamp set after restart", func(ctx context.Context) {
			podName := "terminal-restart-pod" + string(uuid.NewUUID())
			gracePeriod := int64(30)
			podSpec := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					TerminationGracePeriodSeconds: &gracePeriod,
					RestartPolicy:                 v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:    podName,
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"sh", "-c"},
							Args: []string{`
							sleep 9999999 &
							PID=$!

							_term () {
							   kill $PID
							   echo "Caught SIGTERM!"
							}

							trap _term SIGTERM
							touch /tmp/trap-marker

							wait $PID
							trap - TERM

							# Wait for the long running sleep to exit
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
			ginkgo.By(fmt.Sprintf("Creating a pod (%v/%v) with restart policy: %v", f.Namespace.Name, podName, podSpec.Spec.RestartPolicy))
			pod := e2epod.NewPodClient(f).Create(ctx, podSpec)

			ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be running, and with the SIGTERM trap registered", f.Namespace.Name, pod.Name))
			err := e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name, f.Timeouts.PodStart)
			framework.ExpectNoError(err, "Failed to await for the pod to be running: (%v/%v)", f.Namespace.Name, pod.Name)

			w := &cache.ListWatch{
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					return f.ClientSet.CoreV1().Pods(f.Namespace.Name).Watch(ctx, options)
				},
			}

			podsList, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err, "Failed to list pods in namespace: %s", f.Namespace.Name)

			ginkgo.By(fmt.Sprintf("Deleting the pod (%v/%v) to set a deletion timestamp", pod.Namespace, pod.Name))
			err = e2epod.NewPodClient(f).Delete(ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod})
			framework.ExpectNoError(err, "Failed to delete the pod: %q", pod.Name)

			ctxUntil, cancel := context.WithTimeout(ctx, f.Timeouts.PodStart)
			defer cancel()

			ginkgo.By(fmt.Sprintf("Started watch for pod (%v/%v) to enter succeeded phase", pod.Namespace, pod.Name))
			_, err = watchtools.Until(ctxUntil, podsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
				if pod, ok := event.Object.(*v1.Pod); ok {
					found := pod.ObjectMeta.Name == podName &&
						pod.ObjectMeta.Namespace == f.Namespace.Name &&
						pod.Status.Phase == v1.PodSucceeded
					if !found {
						ginkgo.By(fmt.Sprintf("Observed Pod (%s/%s) in phase %v", pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, pod.Status.Phase))
						return false, nil
					}
					ginkgo.By(fmt.Sprintf("Found Pod (%s/%s) in phase %v", pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, pod.Status.Phase))
					return found, nil
				}
				ginkgo.By(fmt.Sprintf("Observed event: %+v", event.Object))
				return false, nil
			})
			ginkgo.By("Ended watch for pod entering succeeded phase")
			framework.ExpectNoError(err, "failed to see event that pod (%s/%s) enter succeeded phase: %v", pod.Namespace, pod.Name, err)

			// As soon as the pod enters succeeded phase (detected by the watch above); kill the kubelet.
			// This is a bit racy, but the goal is to stop the kubelet before the kubelet is able to delete the pod from the API-sever in order to repro https://issues.k8s.io/116925
			ginkgo.By("Stopping the kubelet")
			restartKubelet := mustStopKubelet(ctx, f)

			ginkgo.By("Restarting the kubelet")
			restartKubelet(ctx)

			// Wait for the Kubelet to be ready.
			gomega.Eventually(ctx, func(ctx context.Context) bool {
				nodes, err := e2enode.TotalReady(ctx, f.ClientSet)
				framework.ExpectNoError(err)
				return nodes == 1
			}, time.Minute, f.Timeouts.Poll).Should(gomega.BeTrueBecause("expected kubelet to be in ready state"))

			ginkgo.By(fmt.Sprintf("After the kubelet is restarted, verify the pod (%s/%s) is deleted by kubelet", pod.Namespace, pod.Name))
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodDisappear(ctx, f.ClientSet, pod.Name, pod.Namespace)
			}, f.Timeouts.PodDelete, f.Timeouts.Poll).Should(gomega.BeNil())
		})
		// Regression test for https://issues.k8s.io/118472
		ginkgo.It("should force-delete non-admissible pods created and deleted during kubelet restart", func(ctx context.Context) {
			podName := "rejected-deleted-pod" + string(uuid.NewUUID())
			gracePeriod := int64(30)
			nodeName := getNodeName(ctx, f)
			podSpec := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					NodeName: nodeName,
					NodeSelector: map[string]string{
						"this-label": "does-not-exist-on-any-nodes",
					},
					TerminationGracePeriodSeconds: &gracePeriod,
					RestartPolicy:                 v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:  podName,
							Image: imageutils.GetPauseImageName(),
						},
					},
				},
			})
			ginkgo.By("Stopping the kubelet")
			restartKubelet := mustStopKubelet(ctx, f)

			// Create the pod bound to the node. It will remain in the Pending
			// phase as Kubelet is down.
			ginkgo.By(fmt.Sprintf("Creating a pod (%v/%v)", f.Namespace.Name, podName))
			pod := e2epod.NewPodClient(f).Create(ctx, podSpec)

			ginkgo.By(fmt.Sprintf("Deleting the pod (%v/%v) to set a deletion timestamp", pod.Namespace, pod.Name))
			err := e2epod.NewPodClient(f).Delete(ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod})
			framework.ExpectNoError(err, "Failed to delete the pod: %q", pod.Name)

			// Restart Kubelet so that it proceeds with deletion
			ginkgo.By("Starting the kubelet")
			restartKubelet(ctx)

			ginkgo.By(fmt.Sprintf("After the kubelet is restarted, verify the pod (%v/%v) is deleted by kubelet", pod.Namespace, pod.Name))
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodDisappear(ctx, f.ClientSet, pod.Name, pod.Namespace)
			}, f.Timeouts.PodDelete, f.Timeouts.Poll).Should(gomega.BeNil())
		})
		// Regression test for an extended scenario for https://issues.k8s.io/118472
		ginkgo.It("should force-delete non-admissible pods that was admitted and running before kubelet restart", func(ctx context.Context) {
			nodeLabelKey := "custom-label-key-required"
			nodeLabelValueRequired := "custom-label-value-required-for-admission"
			podName := "rejected-deleted-run" + string(uuid.NewUUID())
			gracePeriod := int64(30)
			nodeName := getNodeName(ctx, f)
			pod := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					NodeSelector: map[string]string{
						nodeLabelKey: nodeLabelValueRequired,
					},
					NodeName:                      nodeName,
					TerminationGracePeriodSeconds: &gracePeriod,
					RestartPolicy:                 v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:  podName,
							Image: imageutils.GetPauseImageName(),
						},
					},
				},
			})

			ginkgo.By(fmt.Sprintf("Adding node label for node (%v) to allow admission of pod (%v/%v)", nodeName, f.Namespace.Name, podName))
			e2enode.AddOrUpdateLabelOnNode(f.ClientSet, nodeName, nodeLabelKey, nodeLabelValueRequired)
			ginkgo.DeferCleanup(func() { e2enode.RemoveLabelOffNode(f.ClientSet, nodeName, nodeLabelKey) })

			// Create the pod bound to the node. It will start, but will be rejected after kubelet restart.
			ginkgo.By(fmt.Sprintf("Creating a pod (%v/%v)", f.Namespace.Name, podName))
			pod = e2epod.NewPodClient(f).Create(ctx, pod)

			ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be running", f.Namespace.Name, pod.Name))
			err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err, "Failed to await for the pod to be running: (%v/%v)", f.Namespace.Name, pod.Name)

			ginkgo.By("Stopping the kubelet")
			restartKubelet := mustStopKubelet(ctx, f)

			ginkgo.By(fmt.Sprintf("Deleting the pod (%v/%v) to set a deletion timestamp", pod.Namespace, pod.Name))
			err = e2epod.NewPodClient(f).Delete(ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod})
			framework.ExpectNoError(err, "Failed to delete the pod: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Removing node label for node (%v) to ensure the pod (%v/%v) is rejected after kubelet restart", nodeName, f.Namespace.Name, podName))
			e2enode.RemoveLabelOffNode(f.ClientSet, nodeName, nodeLabelKey)

			// Restart Kubelet so that it proceeds with deletion
			ginkgo.By("Restarting the kubelet")
			restartKubelet(ctx)

			// Wait for the Kubelet to be ready.
			gomega.Eventually(ctx, func(ctx context.Context) bool {
				nodes, err := e2enode.TotalReady(ctx, f.ClientSet)
				framework.ExpectNoError(err)
				return nodes == 1
			}, time.Minute, f.Timeouts.Poll).Should(gomega.BeTrueBecause("expected kubelet to be in ready state"))

			ginkgo.By(fmt.Sprintf("Once Kubelet is restarted, verify the pod (%v/%v) is deleted by kubelet", pod.Namespace, pod.Name))
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodDisappear(ctx, f.ClientSet, pod.Name, pod.Namespace)
			}, f.Timeouts.PodDelete, f.Timeouts.Poll).Should(gomega.BeNil())
		})
	})

})
