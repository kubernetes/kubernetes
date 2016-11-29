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
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
)

const (
	// podCheckInterval is the interval seconds between pod status checks.
	podCheckInterval = time.Second * 2

	// podDisappearTimeout is the timeout to wait node disappear.
	podDisappearTimeout = time.Minute * 2

	// containerGCPeriod is the period of container garbage collect loop. It should be the same
	// with ContainerGCPeriod in kubelet.go. However we don't want to include kubelet package
	// directly which will introduce a lot more dependencies.
	containerGCPeriod = time.Minute * 1

	dummyFile = "dummy."
)

// TODO: Leverage dynamic Kubelet settings when it's implemented to only modify the kubelet eviction option in this test.
var _ = framework.KubeDescribe("Kubelet Eviction Manager [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("kubelet-eviction-manager")
	var podClient *framework.PodClient
	var c clientset.Interface

	BeforeEach(func() {
		podClient = f.PodClient()
		c = f.ClientSet
	})

	Describe("hard eviction test", func() {
		Context("pod using the most disk space gets evicted when the node disk usage is above the eviction hard threshold", func() {
			var busyPodName, idlePodName, verifyPodName string

			BeforeEach(func() {
				if !isImageSupported() {
					framework.Skipf("test skipped because the image is not supported by the test")
				}
				if !evictionOptionIsSet() {
					framework.Skipf("test skipped because eviction option is not set")
				}

				busyPodName = "to-evict" + string(uuid.NewUUID())
				idlePodName = "idle" + string(uuid.NewUUID())
				verifyPodName = "verify" + string(uuid.NewUUID())
				createIdlePod(idlePodName, podClient)
				podClient.Create(&v1.Pod{
					ObjectMeta: v1.ObjectMeta{
						Name: busyPodName,
					},
					Spec: v1.PodSpec{
						RestartPolicy: v1.RestartPolicyNever,
						Containers: []v1.Container{
							{
								Image: "gcr.io/google_containers/busybox:1.24",
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

			AfterEach(func() {
				if !isImageSupported() || !evictionOptionIsSet() { // Skip the after each
					return
				}
				podClient.DeleteSync(busyPodName, &v1.DeleteOptions{}, podDisappearTimeout)
				podClient.DeleteSync(idlePodName, &v1.DeleteOptions{}, podDisappearTimeout)
				podClient.DeleteSync(verifyPodName, &v1.DeleteOptions{}, podDisappearTimeout)

				// Wait for 2 container gc loop to ensure that the containers are deleted. The containers
				// created in this test consume a lot of disk, we don't want them to trigger disk eviction
				// again after the test.
				time.Sleep(containerGCPeriod * 2)

				if framework.TestContext.PrepullImages {
					// The disk eviction test may cause the prepulled images to be evicted,
					// prepull those images again to ensure this test not affect following tests.
					PrePullAllImages()
				}
			})

			It("should evict the pod using the most disk space [Slow]", func() {
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

						err = verifyPodEviction(podData)
						if err != nil {
							return err
						}

						podData, err = podClient.Get(idlePodName)
						if err != nil {
							return err
						}

						if podData.Status.Phase != v1.PodRunning {
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
						if !nodeHasDiskPressure(f.ClientSet) {
							return fmt.Errorf("expected disk pressure condition is not set")
						}
						nodeDiskPressureCondition = true
						return fmt.Errorf("waiting for node disk pressure condition to be cleared")
					}

					// After eviction happens the pod is evicted so eventually the node disk pressure should be relieved.
					if !podRescheduleable {
						if nodeHasDiskPressure(f.ClientSet) {
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
					if podData.Status.Phase != v1.PodRunning {
						return fmt.Errorf("waiting for the new pod to be running")
					}

					return nil
				}, time.Minute*15 /* based on n1-standard-1 machine type */, podCheckInterval).Should(BeNil())
			})
		})
	})
})

func createIdlePod(podName string, podClient *framework.PodClient) {
	podClient.Create(&v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Image: framework.GetPauseImageNameForHostArch(),
					Name:  podName,
				},
			},
		},
	})
}

func verifyPodEviction(podData *v1.Pod) error {
	if podData.Status.Phase != v1.PodFailed {
		return fmt.Errorf("expected phase to be failed. got %+v", podData.Status.Phase)
	}
	if podData.Status.Reason != "Evicted" {
		return fmt.Errorf("expected failed reason to be evicted. got %+v", podData.Status.Reason)
	}
	return nil
}

func nodeHasDiskPressure(cs clientset.Interface) bool {
	nodeList := framework.GetReadySchedulableNodesOrDie(cs)
	for _, condition := range nodeList.Items[0].Status.Conditions {
		if condition.Type == v1.NodeDiskPressure {
			return condition.Status == v1.ConditionTrue
		}
	}
	return false
}

func evictionOptionIsSet() bool {
	return len(framework.TestContext.KubeletConfig.EvictionHard) > 0
}

// TODO(random-liu): Use OSImage in node status to do the check.
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
