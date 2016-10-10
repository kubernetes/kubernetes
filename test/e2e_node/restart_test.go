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
	"k8s.io/kubernetes/test/e2e/framework"
	"time"

	. "github.com/onsi/ginkgo"
	"k8s.io/kubernetes/pkg/api"
	"os/exec"
)

var _ = framework.KubeDescribe("Restart [Serial] [Slow] [Disruptive]", func() {
	const (
		POD_COUNT             = 100
		POD_CREATION_INTERVAL = 100 * time.Millisecond
		RECOVER_TIMEOUT       = 3 * time.Minute
	)

	f := framework.NewDefaultFramework("restart-test")
	Context("Docker Daemon", func() {
		Context("Network", func() {
			It("should recover from ip leak", func() {
				By("Creating batch pods on node")
				pods := newTestPods(POD_COUNT, framework.GetPauseImageNameForHostArch(), "restart-docker-test")
				createBatchPodWithRateControl(f, pods, POD_CREATION_INTERVAL)
				defer deletePodsSync(f, pods)

				By("Restart docker daemon 3 times and verify pods")
				for i := 0; i < 3; i += 1 {
					By("Restarting Docker Daemon")
					// TODO: Find a uniform way to deal with systemctl/initctl/service operations. #34494
					if stdout, err := exec.Command("sudo", "service", "docker", "restart").CombinedOutput(); err != nil {
						framework.Logf("Failed to trigger docker restart with startup script: %v, stdout: %q", err, string(stdout))
						if stdout, err = exec.Command("sudo", "systemctl", "restart", "docker").CombinedOutput(); err != nil {
							framework.Failf("Failed to trigger docker restart with systemctl: %v, stdout: %q", err, string(stdout))
						}

					}
					// Sleep for 10 seconds for docker
					time.Sleep(10 * time.Second)

					By("Confirm all pods are ready and running")
					success := false
					for start := time.Now(); time.Since(start) < RECOVER_TIMEOUT; time.Sleep(10 * time.Second) {
						podList, err := f.PodClient().List(api.ListOptions{})

						if err != nil {
							framework.Logf("Failed to list pods on node: %v", err)
							continue
						}

						allClear := true
						for _, pod := range podList.Items {
							if clear, err := framework.PodRunningReady(&pod); err != nil || clear != true {
								allClear = false
								framework.Logf("Pod %q/%q is not running and ready. Keep Waiting.", pod.Namespace, pod.Name)
								break
							}
						}
						if allClear == true {
							success = true
							break
						}
					}

					if !success {
						framework.Failf("After restarting docker daemon, not all pods are ready and running.")
					}
				}
			})
		})
	})
})
