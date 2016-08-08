/*
Copyright 2016 The Kubernetes Authors.

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
	"os/exec"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

const (
	// podCheckInterval is the interval seconds between pod status checks.
	podCheckInterval = time.Second * 2

	dummyFile = "dummy."
)

// TODO: Leverage dynamic Kubelet settings when it's implemented to only modify the kubelet eviction option in this test.
// To manually trigger the test on a node with disk space just over 15Gi :
//   make test-e2e-node FOCUS="hard eviction test" TEST_ARGS="--eviction-hard=nodefs.available<15Gi"
var _ = framework.KubeDescribe("Kubelet Eviction Manager [Flaky] [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("kubelet-eviction-manager")
	var podClient *framework.PodClient
	var c *client.Client
	var n *api.Node

	BeforeEach(func() {
		podClient = f.PodClient()
		c = f.Client
		nodeList := framework.GetReadySchedulableNodesOrDie(c)
		n = &nodeList.Items[0]
	})

	Describe("hard eviction test", func() {
		Context("pod using the most disk space gets evicted when the node disk usage is above the eviction hard threshold", func() {
			var busyPodName, idlePodName string
			var containersToCleanUp map[string]bool

			AfterEach(func() {
				podClient.Delete(busyPodName, &api.DeleteOptions{})
				podClient.Delete(idlePodName, &api.DeleteOptions{})
				for container := range containersToCleanUp {
					// TODO: to be container implementation agnostic
					cmd := exec.Command("docker", "rm", "-f", strings.Trim(container, dockertools.DockerPrefix))
					cmd.Run()
				}
			})

			BeforeEach(func() {
				if !evictionOptionIsSet() {
					return
				}

				busyPodName = "to-evict" + string(uuid.NewUUID())
				idlePodName = "idle" + string(uuid.NewUUID())
				containersToCleanUp = make(map[string]bool)
				podClient.Create(&api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name: idlePodName,
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyNever,
						Containers: []api.Container{
							{
								Image: ImageRegistry[pauseImage],
								Name:  idlePodName,
							},
						},
					},
				})
				podClient.Create(&api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name: busyPodName,
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyNever,
						Containers: []api.Container{
							{
								Image: ImageRegistry[busyBoxImage],
								Name:  busyPodName,
								// Filling the disk
								Command: []string{"sh", "-c",
									fmt.Sprintf("for NUM in `seq 1 1 1000`; do dd if=/dev/urandom of=%s.$NUM bs=4000000 count=10; sleep 3; done",
										dummyFile)},
							},
						},
					},
				})
			})

			It("should evict the pod using the most disk space", func() {
				if !evictionOptionIsSet() {
					framework.Logf("test skipped because eviction option is not set")
					return
				}

				evictionOccurred := false
				Eventually(func() error {
					if !evictionOccurred {
						podData, err := podClient.Get(busyPodName)
						if err != nil {
							return err
						}
						recordContainerId(containersToCleanUp, podData.Status.ContainerStatuses)

						err = verifyPodEviction(podData)
						if err != nil {
							return err
						}
						if !nodeHasDiskPressure(f.Client) {
							return fmt.Errorf("expected disk pressure condition is not set")
						}

						podData, err = podClient.Get(idlePodName)
						if err != nil {
							return err
						}
						recordContainerId(containersToCleanUp, podData.Status.ContainerStatuses)

						if podData.Status.Phase != api.PodRunning {
							return fmt.Errorf("expected phase to be running. got %+v", podData.Status.Phase)
						}

						evictionOccurred = true
					}

					// After eviction happens the pod is evicted so eventually the node disk pressure should be gone.
					if nodeHasDiskPressure(f.Client) {
						return fmt.Errorf("expected disk pressure condition relief has not happened")
					}
					return nil
				}, time.Minute*5, podCheckInterval).Should(BeNil())
			})
		})
	})
})

func verifyPodEviction(podData *api.Pod) error {
	if podData.Status.Phase != api.PodFailed {
		return fmt.Errorf("expected phase to be failed. got %+v", podData.Status.Phase)
	}
	if podData.Status.Reason != "Evicted" {
		return fmt.Errorf("expected failed reason to be evicted. got %+v", podData.Status.Reason)
	}
	return nil
}

func nodeHasDiskPressure(c *client.Client) bool {
	nodeList := framework.GetReadySchedulableNodesOrDie(c)
	for _, condition := range nodeList.Items[0].Status.Conditions {
		if condition.Type == api.NodeDiskPressure {
			return condition.Status == api.ConditionTrue
		}
	}
	return false
}

func recordContainerId(containersToCleanUp map[string]bool, containerStatuses []api.ContainerStatus) {
	for _, status := range containerStatuses {
		containersToCleanUp[status.ContainerID] = true
	}
}

func evictionOptionIsSet() bool {
	return len(framework.TestContext.EvictionHard) > 0
}
