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
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	apimetav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/json"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"
)

// waitForPods waits for timeout duration, for pod_count.
// If the timeout is hit, it returns the list of currently running pods.
func waitForPods(f *framework.Framework, pod_count int, timeout time.Duration) (runningPods []*v1.Pod) {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(10 * time.Second) {
		podList, err := f.PodClient().List(metav1.ListOptions{})
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

type DockerDaemonOption struct {
	Live_restore bool `json:"live-restore"`
}

func getDockerVersion() (string, error) {
	c, err := cadvisor.New(0 /*don't start the http server*/, "docker", "/var/lib/kubelet")
	if err != nil {
		return "", fmt.Errorf("Could not start cadvisor %v", err)
	}

	vi, err := c.VersionInfo()
	if err != nil {
		return "", fmt.Errorf("Could not get VersionInfo %v", err)
	}
	return vi.DockerVersion, nil
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

				pods := newTestPods(podCount, false, framework.GetPauseImageNameForHostArch(), "restart-docker-test")
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
				framework.SkipIfContainerRuntimeIs("rkt")

				// Skip when docker engine version is less than 1.12
				dockerVersion, err := getDockerVersion()
				dockerRegex, _ := regexp.Compile(`1\.12\.[0-9]+`)
				if !dockerRegex.Match([]byte(dockerVersion)) {
					framework.Skipf("Live restore isn't suported before Docker Engine version 1.12")
				}

				// skip when docker live-restore is not enabled
				dockerPid, err := ioutil.ReadFile("/var/run/docker.pid")
				framework.ExpectNoError(err, "Failed to get docker daemon PID: %v, stdout: %q", err, string(dockerPid))

				dockerdCmd, _ := ioutil.ReadFile("/proc/" + string(dockerPid) + "/cmdline")

				daemonConfFile := "/etc/docker/daemon.json"
				_, err = os.Stat(daemonConfFile)
				var inConfFile bool
				if !os.IsNotExist(err) {
					opt := new(DockerDaemonOption)
					data, err := ioutil.ReadFile(daemonConfFile)
					err = json.Unmarshal(data, &opt)
					framework.ExpectNoError(err, "Failed to parse docker config file %q: %v", daemonConfFile, err)
					inConfFile = opt.Live_restore
				}
				asOption := strings.Contains(string(dockerdCmd), "--live-restore")

				if !asOption && !inConfFile {
					framework.Skipf("Docker live restore is not set")
				}

				// create pod
				podClient := f.PodClient()
				name := "pod-test-live-restore"
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

				// ensure the pod has the same container
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
