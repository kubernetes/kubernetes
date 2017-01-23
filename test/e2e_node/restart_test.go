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
	"io/ioutil"
	apimetav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/uuid"
	testutils "k8s.io/kubernetes/test/utils"
	"os"
	"reflect"
	"strconv"
	"strings"
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

func dockerReloadConfiguration(f *framework.Framework) {
	//get docker PID
	dockerPid, err := ioutil.ReadFile("/var/run/docker.pid")
	framework.ExpectNoError(err, "Failed to get docker daemon PID: %v, stdout: %q", err, string(dockerPid))

	// send SIGHUP to docker
	stdout, err := exec.Command("sudo", "kill", "-s", "SIGHUP", string(dockerPid)).CombinedOutput()
	framework.ExpectNoError(err, "Failed to send SIGHUP signal to dockerd-current: %v, stdout: %q", err, string(stdout))
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
		Context("Live restore", func() {
			It("should restore container when live restore is on", func() {

				// ensure live-restore is on
				dockerPid, err := ioutil.ReadFile("/var/run/docker.pid")
				framework.ExpectNoError(err, "Failed to get docker daemon PID: %v, stdout: %q", err, string(dockerPid))

				dockerdCmd, _ := ioutil.ReadFile("/proc/" + string(dockerPid) + "/cmdline")
				if !strings.Contains(string(dockerdCmd), "--live-restore") {
					confFilePath := "/etc/docker/daemon.json"
					if _, err := os.Stat(confFilePath); os.IsNotExist(err) {
						ioutil.WriteFile(confFilePath, []byte("{\n\"live-restore\": true\n}\n"), 0644)
						dockerReloadConfiguration(f)
						defer func() {
							os.Remove(confFilePath)
							dockerReloadConfiguration(f)
						}()
					} else {
						confCopy, err := ioutil.ReadFile(confFilePath)
						framework.ExpectNoError(err, "Failed to read docker config file: %v", confCopy)

						os.Remove(confFilePath)
						ioutil.WriteFile(confFilePath, []byte("{\n\"live-restore\": true\n}\n"), 0644)

						dockerReloadConfiguration(f)
						defer func() {
							os.Remove(confFilePath)
							ioutil.WriteFile(confFilePath, confCopy, 0644)
							dockerReloadConfiguration(f)
						}()
					}
				}

				// create pod
				podClient := f.PodClient()
				name := "pod-" + string(uuid.NewUUID())
				value := strconv.Itoa(time.Now().Nanosecond())
				pod := &v1.Pod{
					ObjectMeta: apimetav1.ObjectMeta{
						Name: name,
						Labels: map[string]string{
							"name": "foo",
							"time": value,
						},
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name:  "nginx",
								Image: "gcr.io/google_containers/nginx-slim:0.7",
							},
						},
					},
				}

				preRestartPod := podClient.CreateSync(pod)
				By(fmt.Sprintf("Pre restart container id %q", preRestartPod.Status.ContainerStatuses[0].ContainerID))

				// restart docker
				if stdout, err := exec.Command("sudo", "systemctl", "restart", "docker").CombinedOutput(); err != nil {
					framework.Logf("Failed to trigger docker restart with systemd/systemctl: %v, stdout: %q", err, string(stdout))
					if stdout, err = exec.Command("sudo", "service", "docker", "restart").CombinedOutput(); err != nil {
						framework.Failf("Failed to trigger docker restart with upstart/service: %v, stdout: %q", err, string(stdout))
					}
				}

				// assure the pod has the same container
				postRestartRunningPods := waitForPods(f, 1, recoverTimeout)
				if len(postRestartRunningPods) == 0 {
					framework.Failf("Failed to start *any* pods after docker restart")

				}
				postRestartPod, err := podClient.Get(preRestartPod.Name, apimetav1.GetOptions{})
				framework.ExpectNoError(err, "Could not get pod after restart")

				if !reflect.DeepEqual(preRestartPod.Status.ContainerStatuses, postRestartPod.Status.ContainerStatuses) {
					framework.Failf("Containers statuses has modified after docker restart\n pre-restart: %v post-restart\n: %v, ", preRestartPod.Status.ContainerStatuses, postRestartPod.Status.ContainerStatuses)
				}
				if postRestartPod == nil {
					framework.Failf("Pod not found after restart")
				}
				if len(postRestartPod.Status.ContainerStatuses) == 0 {
					framework.Failf("No containers found on pod after restart")
				}
				By(fmt.Sprintf("Post restart container id %q", postRestartPod.Status.ContainerStatuses[0].ContainerID))
			})
		})
	})
})
