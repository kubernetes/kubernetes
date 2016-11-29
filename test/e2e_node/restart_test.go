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

package e2e_node

import (
	"time"

	"k8s.io/kubernetes/test/e2e/framework"

	"fmt"
	"os/exec"

	. "github.com/onsi/ginkgo"
	"k8s.io/kubernetes/pkg/api/v1"
	testutils "k8s.io/kubernetes/test/utils"
)

// waitForPods waits for timeout duration, for pod_count.
// If the timeout is hit, it returns the list of currently running pods.
func waitForPods(f *framework.Framework, pod_count int, timeout time.Duration) (runningPods []*v1.Pod) {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(10 * time.Second) {
		podList, err := f.PodClient().List(v1.ListOptions{})
		if err != nil {
			framework.Logf("Failed to list pods on node: %v", err)
			continue
		}

		runningPods = []*v1.Pod{}
		for _, pod := range podList.Items {
			if r, err := testutils.PodRunningReady(&pod); err != nil || !r {
				continue
			}
			runningPods = append(runningPods, &pod)
		}
		framework.Logf("Running pod count %d", len(runningPods))
		if len(runningPods) >= pod_count {
			break
		}
	}
	return runningPods
}

var _ = framework.KubeDescribe("Restart [Serial] [Slow] [Disruptive]", func() {
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
	Context("Docker Daemon", func() {
		Context("Network", func() {
			It("should recover from ip leak", func() {

				pods := newTestPods(podCount, framework.GetPauseImageNameForHostArch(), "restart-docker-test")
				By(fmt.Sprintf("Trying to create %d pods on node", len(pods)))
				createBatchPodWithRateControl(f, pods, podCreationInterval)
				defer deletePodsSync(f, pods)

				// Give the node some time to stabilize, assume pods that enter RunningReady within
				// startTimeout fit on the node and the node is now saturated.
				runningPods := waitForPods(f, podCount, startTimeout)
				if len(runningPods) < minPods {
					framework.Failf("Failed to start %d pods, cannot test that restarting docker doesn't leak IPs", minPods)
				}

				for i := 0; i < restartCount; i += 1 {
					By(fmt.Sprintf("Restarting Docker Daemon iteration %d", i))

					// TODO: Find a uniform way to deal with systemctl/initctl/service operations. #34494
					if stdout, err := exec.Command("sudo", "systemctl", "restart", "docker").CombinedOutput(); err != nil {
						framework.Logf("Failed to trigger docker restart with systemd/systemctl: %v, stdout: %q", err, string(stdout))
						if stdout, err = exec.Command("sudo", "service", "docker", "restart").CombinedOutput(); err != nil {
							framework.Failf("Failed to trigger docker restart with upstart/service: %v, stdout: %q", err, string(stdout))
						}
					}
					time.Sleep(20 * time.Second)
				}

				By("Checking currently Running/Ready pods")
				postRestartRunningPods := waitForPods(f, len(runningPods), recoverTimeout)
				if len(postRestartRunningPods) == 0 {
					framework.Failf("Failed to start *any* pods after docker restart, this might indicate an IP leak")
				}
				By("Confirm no containers have terminated")
				for _, pod := range postRestartRunningPods {
					if c := testutils.TerminatedContainers(pod); len(c) != 0 {
						framework.Failf("Pod %q has failed containers %+v after docker restart, this might indicate an IP leak", pod.Name, c)
					}
				}
				By(fmt.Sprintf("Docker restart test passed with %d pods", len(postRestartRunningPods)))
			})
		})
	})
})
