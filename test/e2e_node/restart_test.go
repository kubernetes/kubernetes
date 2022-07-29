//go:build linux
// +build linux

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
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

// waitForPods waits for timeout duration, for podCount.
// If the timeout is hit, it returns the list of currently running pods.
func waitForPods(f *framework.Framework, podCount int, timeout time.Duration) (runningPods []*v1.Pod) {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(10 * time.Second) {
		podList, err := f.PodClient().List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			framework.Logf("Failed to list pods on node: %v", err)
			continue
		}

		runningPods = []*v1.Pod{}
		for i := range podList.Items {
			pod := podList.Items[i]
			if r, err := testutils.PodRunningReadyOrSucceeded(&pod); err != nil || !r {
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

var _ = SIGDescribe("Restart [Serial] [Slow] [Disruptive]", func() {
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
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	ginkgo.Context("Container Runtime", func() {
		ginkgo.Context("Network", func() {
			ginkgo.It("should recover from ip leak", func() {
				pods := newTestPods(podCount, false, imageutils.GetPauseImageName(), "restart-container-runtime-test")
				ginkgo.By(fmt.Sprintf("Trying to create %d pods on node", len(pods)))
				createBatchPodWithRateControl(f, pods, podCreationInterval)
				defer deletePodsSync(f, pods)

				// Give the node some time to stabilize, assume pods that enter RunningReady within
				// startTimeout fit on the node and the node is now saturated.
				runningPods := waitForPods(f, podCount, startTimeout)
				if len(runningPods) < minPods {
					framework.Failf("Failed to start %d pods, cannot test that restarting container runtime doesn't leak IPs", minPods)
				}

				for i := 0; i < restartCount; i++ {
					ginkgo.By(fmt.Sprintf("Killing container runtime iteration %d", i))
					// Wait for container runtime to be running
					var pid int
					gomega.Eventually(func() error {
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
				postRestartRunningPods := waitForPods(f, len(runningPods), recoverTimeout)
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
		ginkgo.It("should correctly account for terminated pods after restart", func() {
			node := getLocalNode(f)
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
			createBatchPodWithRateControl(f, restartNeverPods, podCreationInterval)
			defer deletePodsSync(f, restartNeverPods)

			completedPods := waitForPods(f, podCountRestartNever, startTimeout)
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
			createBatchPodWithRateControl(f, restartAlwaysPods, podCreationInterval)
			defer deletePodsSync(f, restartAlwaysPods)

			numAllPods := podCountRestartNever + podCountRestartAlways
			allPods := waitForPods(f, numAllPods, startTimeout)
			if len(allPods) < numAllPods {
				framework.Failf("Failed to run sufficient restartAlways pods, got %d but expected %d", len(allPods), numAllPods)
			}

			ginkgo.By("killing and restarting kubelet")
			// We want to kill the kubelet rather than a graceful restart
			startKubelet := stopKubelet()
			startKubelet()

			// If this test works correctly, each of these pods will exit
			// with no issue. But if accounting breaks, pods scheduled after
			// restart may think these old pods are consuming CPU and we
			// will get an OutOfCpu error.
			ginkgo.By("verifying restartNever pods succeed and restartAlways pods stay running")
			for start := time.Now(); time.Since(start) < startTimeout; time.Sleep(10 * time.Second) {
				postRestartRunningPods := waitForPods(f, numAllPods, recoverTimeout)
				if len(postRestartRunningPods) < numAllPods {
					framework.Failf("less pods are running after node restart, got %d but expected %d", len(postRestartRunningPods), numAllPods)
				}
			}
		})
	})
})
