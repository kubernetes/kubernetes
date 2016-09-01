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
var _ = framework.KubeDescribe("Kubelet Eviction Manager [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("kubelet-eviction-manager")
	var podClient *framework.PodClient
	var c *client.Client

	BeforeEach(func() {
		podClient = f.PodClient()
		c = f.Client
	})

	Describe("hard eviction test", func() {
		Context("pod using the most disk space gets evicted when the node disk usage is above the eviction hard threshold", func() {
			var busyPodName, idlePodName, verifyPodName string
			var containersToCleanUp map[string]bool

			AfterEach(func() {
				podClient.Delete(busyPodName, &api.DeleteOptions{})
				podClient.Delete(idlePodName, &api.DeleteOptions{})
				podClient.Delete(verifyPodName, &api.DeleteOptions{})
				for container := range containersToCleanUp {
					// TODO: to be container implementation agnostic
					cmd := exec.Command("docker", "rm", "-f", strings.Trim(container, dockertools.DockerPrefix))
					cmd.Run()
				}
			})

			BeforeEach(func() {
				if !isImageSupported() || !evictionOptionIsSet() {
					return
				}

				busyPodName = "to-evict" + string(uuid.NewUUID())
				idlePodName = "idle" + string(uuid.NewUUID())
				verifyPodName = "verify" + string(uuid.NewUUID())
				containersToCleanUp = make(map[string]bool)
				createIdlePod(idlePodName, podClient)
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
									fmt.Sprintf("for NUM in `seq 1 1 100000`; do dd if=/dev/urandom of=%s.$NUM bs=50000000 count=10; sleep 0.5; done",
										dummyFile)},
							},
						},
					},
				})
			})

			It("should evict the pod using the most disk space [Slow]", func() {
				if !isImageSupported() {
					framework.Logf("test skipped because the image is not supported by the test")
					return
				}
				if !evictionOptionIsSet() {
					framework.Logf("test skipped because eviction option is not set")
					return
				}

				evictionOccurred := false
				nodeDiskPressureCondition := false
				podRescheduleable := false
				Eventually(func() error {
					// Avoid the test using up all the disk space
					err := checkDiskUsage(0.05)
					if err != nil {
						return err
					}

					// The pod should be evicted.
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

						podData, err = podClient.Get(idlePodName)
						if err != nil {
							return err
						}
						recordContainerId(containersToCleanUp, podData.Status.ContainerStatuses)

						if podData.Status.Phase != api.PodRunning {
							err = verifyPodEviction(podData)
							if err != nil {
								return err
							}
						}
						evictionOccurred = true
						return fmt.Errorf("waiting for node disk pressure condition to be set")
					}

					// The node should have disk pressure condition after the pods are evicted.
					if !nodeDiskPressureCondition {
						if !nodeHasDiskPressure(f.Client) {
							return fmt.Errorf("expected disk pressure condition is not set")
						}
						nodeDiskPressureCondition = true
						return fmt.Errorf("waiting for node disk pressure condition to be cleared")
					}

					// After eviction happens the pod is evicted so eventually the node disk pressure should be relieved.
					if !podRescheduleable {
						if nodeHasDiskPressure(f.Client) {
							return fmt.Errorf("expected disk pressure condition relief has not happened")
						}
						createIdlePod(verifyPodName, podClient)
						podRescheduleable = true
						return fmt.Errorf("waiting for the node to accept a new pod")
					}

					// The new pod should be able to be scheduled and run after the disk pressure is relieved.
					podData, err := podClient.Get(verifyPodName)
					if err != nil {
						return err
					}
					recordContainerId(containersToCleanUp, podData.Status.ContainerStatuses)
					if podData.Status.Phase != api.PodRunning {
						return fmt.Errorf("waiting for the new pod to be running")
					}

					return nil
				}, time.Minute*15 /* based on n1-standard-1 machine type */, podCheckInterval).Should(BeNil())
			})
		})
	})
})

func createIdlePod(podName string, podClient *framework.PodClient) {
	podClient.Create(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: podName,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyNever,
			Containers: []api.Container{
				{
					Image: ImageRegistry[pauseImage],
					Name:  podName,
				},
			},
		},
	})
}

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

func isImageSupported() bool {
	// TODO: Only images with image fs is selected for testing for now. When the kubelet settings can be dynamically updated,
	// instead of skipping images the eviction thresholds should be adjusted based on the images.
	return strings.Contains(framework.TestContext.NodeName, "-gci-dev-")
}

// checkDiskUsage verifies that the available bytes on disk are above the limit.
func checkDiskUsage(limit float64) error {
	summary, err := getNodeSummary()
	if err != nil {
		return err
	}

	if nodeFs := summary.Node.Fs; nodeFs != nil {
		if nodeFs.AvailableBytes != nil && nodeFs.CapacityBytes != nil {
			if float64(*nodeFs.CapacityBytes)*limit > float64(*nodeFs.AvailableBytes) {
				return fmt.Errorf("available nodefs byte is less than %v%%", limit*float64(100))
			}
		}
	}

	if summary.Node.Runtime != nil {
		if imageFs := summary.Node.Runtime.ImageFs; imageFs != nil {
			if float64(*imageFs.CapacityBytes)*limit > float64(*imageFs.AvailableBytes) {
				return fmt.Errorf("available imagefs byte is less than %v%%", limit*float64(100))
			}
		}
	}

	return nil
}
