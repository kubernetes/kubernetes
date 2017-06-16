/*
Copyright 2017 The Kubernetes Authors.

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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	nodeutil "k8s.io/kubernetes/pkg/api/v1/node"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Eviction Policy is described here:
// https://github.com/kubernetes/kubernetes/blob/master/docs/proposals/kubelet-eviction.md

var _ = framework.KubeDescribe("LocalStorageAllocatableEviction [Slow] [Serial] [Disruptive] [Flaky]", func() {
	f := framework.NewDefaultFramework("localstorageallocatable-eviction-test")
	evictionTestTimeout := 15 * time.Minute
	testCondition := "Evict pod due to local storage allocatable violation"
	conditionType := v1.NodeDiskPressure
	var podTestSpecs []podTestSpec
	//podTestSpecsS := make([]podTestSpec, 5)
	var diskReserve uint64
	Context(fmt.Sprintf("when we run containers that should cause %s", testCondition), func() {

		BeforeEach(func() {
			diskAvail, err := getDiskUsage()
			if err != nil {
				framework.ExpectNoError(err)
			}

			diskReserve = uint64(0.8 * diskAvail / 1000000) // Reserve 0.8 * disk Capacity for kube-reserved scratch storage
			maxDisk := 10000000                             // Set dd command to read and write up to 10MB at a time
			count := uint64(0.8 * diskAvail / float64(maxDisk))
			command := fmt.Sprintf("dd if=/dev/urandom of=dummy bs=%d count=%d; sleep 0.5; while true; do sleep 5; done", maxDisk, count)

			podTestSpecs = []podTestSpec{
				{
					evictionPriority: 1, // This pod should be evicted before the innocent pod
					pod: &v1.Pod{
						ObjectMeta: metav1.ObjectMeta{Name: "container-disk-hog-pod"},
						Spec: v1.PodSpec{
							RestartPolicy: v1.RestartPolicyNever,
							Containers: []v1.Container{
								{
									Image:   "gcr.io/google_containers/busybox:1.24",
									Name:    "container-disk-hog-pod",
									Command: []string{"sh", "-c", command},
								},
							},
						},
					},
				},

				{
					evictionPriority: 0, // This pod should never be evicted
					pod: &v1.Pod{
						ObjectMeta: metav1.ObjectMeta{Name: "idle-pod"},
						Spec: v1.PodSpec{
							RestartPolicy: v1.RestartPolicyNever,
							Containers: []v1.Container{
								{
									Image: "gcr.io/google_containers/busybox:1.24",
									Name:  "idle-pod",
									Command: []string{"sh", "-c",
										fmt.Sprintf("while true; do sleep 5; done")},
								},
							},
						},
					},
				},
			}
		})

		// Set up --kube-reserved for scratch storage
		tempSetCurrentKubeletConfig(f, func(initialConfig *componentconfig.KubeletConfiguration) {
			framework.Logf("Set up --kube-reserved for local storage reserved %dMi", diskReserve)
			initialConfig.KubeReserved = componentconfig.ConfigurationMap(map[string]string{"storage": fmt.Sprintf("%dMi", diskReserve)})

		})

		// Place the remainder of the test within a context so that the kubelet config is set before and after the test.
		Context("With kubeconfig updated", func() {
			runLocalStorageEvictionTest(f, conditionType, testCondition, &podTestSpecs, evictionTestTimeout, hasDiskPressure)
		})

	})

})

// Returns TRUE if the node has disk pressure, FALSE otherwise
func hasDiskPressure(f *framework.Framework, conditionType v1.NodeConditionType, testCondition string) (bool, error) {
	localNodeStatus := getLocalNode(f).Status
	_, pressure := nodeutil.GetNodeCondition(&localNodeStatus, conditionType)
	Expect(pressure).NotTo(BeNil())
	hasPressure := pressure.Status == v1.ConditionTrue
	By(fmt.Sprintf("checking if pod has %s: %v", testCondition, hasPressure))

	// Additional Logging relating to disk
	summary, err := getNodeSummary()
	if err != nil {
		return false, err
	}
	if summary.Node.Runtime != nil && summary.Node.Runtime.ImageFs != nil && summary.Node.Runtime.ImageFs.UsedBytes != nil {
		framework.Logf("imageFsInfo.UsedBytes: %d", *summary.Node.Runtime.ImageFs.UsedBytes)
	}
	if summary.Node.Fs != nil && summary.Node.Fs.UsedBytes != nil {
		framework.Logf("rootFsInfo.UsedBytes: %d", *summary.Node.Fs.UsedBytes)
	}
	for _, pod := range summary.Pods {
		framework.Logf("Pod: %s", pod.PodRef.Name)
		for _, container := range pod.Containers {
			if container.Rootfs != nil && container.Rootfs.UsedBytes != nil {
				framework.Logf("--- summary Container: %s UsedBytes: %d", container.Name, *container.Rootfs.UsedBytes)
			}
		}
		for _, volume := range pod.VolumeStats {
			if volume.FsStats.UsedBytes != nil {
				framework.Logf("--- summary Volume: %s UsedBytes: %d", volume.Name, *volume.FsStats.UsedBytes)
			}
		}
	}
	return hasPressure, nil
}

// Pass podTestSpecsP as references so that it could be set up in the first BeforeEach clause
func runLocalStorageEvictionTest(f *framework.Framework, conditionType v1.NodeConditionType, testCondition string, podTestSpecsP *[]podTestSpec, evictionTestTimeout time.Duration,
	hasPressureCondition func(*framework.Framework, v1.NodeConditionType, string) (bool, error)) {
	BeforeEach(func() {

		By("seting up pods to be used by tests")
		for _, spec := range *podTestSpecsP {
			By(fmt.Sprintf("creating pod with container: %s", spec.pod.Name))
			f.PodClient().CreateSync(spec.pod)
		}
	})

	It(fmt.Sprintf("should eventually see %s, and then evict all of the correct pods", testCondition), func() {
		Expect(podTestSpecsP).NotTo(BeNil())
		podTestSpecs := *podTestSpecsP

		Eventually(func() error {
			hasPressure, err := hasPressureCondition(f, conditionType, testCondition)
			if err != nil {
				return err
			}
			if hasPressure {
				return nil
			}
			return fmt.Errorf("Condition: %s not encountered", testCondition)
		}, evictionTestTimeout, evictionPollInterval).Should(BeNil())

		Eventually(func() error {
			// Gather current information
			updatedPodList, err := f.ClientSet.Core().Pods(f.Namespace.Name).List(metav1.ListOptions{})
			updatedPods := updatedPodList.Items
			for _, p := range updatedPods {
				framework.Logf("fetching pod %s; phase= %v", p.Name, p.Status.Phase)
			}
			_, err = hasPressureCondition(f, conditionType, testCondition)
			if err != nil {
				return err
			}

			By("checking eviction ordering and ensuring important pods dont fail")
			done := true
			for _, priorityPodSpec := range podTestSpecs {
				var priorityPod v1.Pod
				for _, p := range updatedPods {
					if p.Name == priorityPodSpec.pod.Name {
						priorityPod = p
					}
				}
				Expect(priorityPod).NotTo(BeNil())

				// Check eviction ordering.
				// Note: it is alright for a priority 1 and priority 2 pod (for example) to fail in the same round
				for _, lowPriorityPodSpec := range podTestSpecs {
					var lowPriorityPod v1.Pod
					for _, p := range updatedPods {
						if p.Name == lowPriorityPodSpec.pod.Name {
							lowPriorityPod = p
						}
					}
					Expect(lowPriorityPod).NotTo(BeNil())
					if priorityPodSpec.evictionPriority < lowPriorityPodSpec.evictionPriority && lowPriorityPod.Status.Phase == v1.PodRunning {
						Expect(priorityPod.Status.Phase).NotTo(Equal(v1.PodFailed),
							fmt.Sprintf("%s pod failed before %s pod", priorityPodSpec.pod.Name, lowPriorityPodSpec.pod.Name))
					}
				}

				// EvictionPriority 0 pods should not fail
				if priorityPodSpec.evictionPriority == 0 {
					Expect(priorityPod.Status.Phase).NotTo(Equal(v1.PodFailed),
						fmt.Sprintf("%s pod failed (and shouldn't have failed)", priorityPod.Name))
				}

				// If a pod that is not evictionPriority 0 has not been evicted, we are not done
				if priorityPodSpec.evictionPriority != 0 && priorityPod.Status.Phase != v1.PodFailed {
					done = false
				}
			}
			if done {
				return nil
			}
			return fmt.Errorf("pods that caused %s have not been evicted.", testCondition)
		}, evictionTestTimeout, evictionPollInterval).Should(BeNil())

		// We observe pressure from the API server.  The eviction manager observes pressure from the kubelet internal stats.
		// This means the eviction manager will observe pressure before we will, creating a delay between when the eviction manager
		// evicts a pod, and when we observe the pressure by querrying the API server.  Add a delay here to account for this delay
		By("making sure pressure from test has surfaced before continuing")
		time.Sleep(pressureDelay)

		By("making sure conditions eventually return to normal")
		Eventually(func() error {
			hasPressure, err := hasPressureCondition(f, conditionType, testCondition)
			if err != nil {
				return err
			}
			if hasPressure {
				return fmt.Errorf("Conditions havent returned to normal, we still have %s", testCondition)
			}
			return nil
		}, evictionTestTimeout, evictionPollInterval).Should(BeNil())

		By("making sure conditions do not return, and that pods that shouldnt fail dont fail")
		Consistently(func() error {
			hasPressure, err := hasPressureCondition(f, conditionType, testCondition)
			if err != nil {
				// Race conditions sometimes occur when checking pressure condition due to #38710 (Docker bug)
				// Do not fail the test when this occurs, since this is expected to happen occasionally.
				framework.Logf("Failed to check pressure condition. Error: %v", err)
				return nil
			}
			if hasPressure {
				return fmt.Errorf("%s dissappeared and then reappeared", testCondition)
			}
			// Gather current information
			updatedPodList, _ := f.ClientSet.Core().Pods(f.Namespace.Name).List(metav1.ListOptions{})
			for _, priorityPodSpec := range podTestSpecs {
				// EvictionPriority 0 pods should not fail
				if priorityPodSpec.evictionPriority == 0 {
					for _, p := range updatedPodList.Items {
						if p.Name == priorityPodSpec.pod.Name && p.Status.Phase == v1.PodFailed {
							return fmt.Errorf("%s pod failed (delayed) and shouldn't have failed", p.Name)
						}
					}
				}
			}
			return nil
		}, postTestConditionMonitoringPeriod, evictionPollInterval).Should(BeNil())

		By("making sure we can start a new pod after the test")
		podName := "test-admit-pod"
		f.PodClient().CreateSync(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
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
	})

	AfterEach(func() {
		By("deleting pods")
		for _, spec := range *podTestSpecsP {
			By(fmt.Sprintf("deleting pod: %s", spec.pod.Name))
			f.PodClient().DeleteSync(spec.pod.Name, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
		}

		if CurrentGinkgoTestDescription().Failed {
			if framework.TestContext.DumpLogsOnFailure {
				logPodEvents(f)
				logNodeEvents(f)
			}
			By("sleeping to allow for cleanup of test")
			time.Sleep(postTestConditionMonitoringPeriod)
		}
	})
}

func getDiskUsage() (float64, error) {
	summary, err := getNodeSummary()
	if err != nil {
		return 0, err
	}

	if nodeFs := summary.Node.Fs; nodeFs != nil {
		return float64(*nodeFs.AvailableBytes), nil
	}

	return 0, fmt.Errorf("fail to get nodefs available bytes")

}
