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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Eviction Policy is described here:
// https://github.com/kubernetes/kubernetes/blob/master/docs/proposals/kubelet-eviction.md

const (
	postTestConditionMonitoringPeriod = 2 * time.Minute
	evictionPollInterval              = 5 * time.Second
)

var _ = framework.KubeDescribe("InodeEviction [Slow] [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("inode-eviction-test")

	podTestSpecs := []podTestSpec{
		{
			evictionPriority: 1, // This pod should be evicted before the normal memory usage pod
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "container-inode-hog-pod"},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image:           "gcr.io/google_containers/busybox:1.24",
							ImagePullPolicy: "Always",
							Name:            "container-inode-hog-pod",
							Command: []string{
								"sh",
								"-c", // Make 1 billion small files (more than we have inodes)
								"for A in `seq 1 1 10000`; do for B in `seq 1 1 100000`; do touch smallfile$A.$B.txt; sleep 0.001; done; done;",
								// If getting out of memory (OOM) errors, `seq 1 1 xxxxxxxx` may not fit in memory
							},
						},
					},
				},
			},
		},
		{
			evictionPriority: 1, // This pod should be evicted before the normal memory usage pod
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "volume-inode-hog-pod"},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image:           "gcr.io/google_containers/busybox:1.24",
							ImagePullPolicy: "Always",
							Name:            "volume-inode-hog-pod",
							Command: []string{
								"sh",
								"-c", // Make 1 billion small files (more than we have inodes)
								"for A in `seq 1 1 10000`; do for B in `seq 1 1 100000`; do touch /test-empty-dir-mnt/smallfile$A.$B.txt; sleep 0.001; done; done;",
								// If getting out of memory (OOM) errors, `seq 1 1 xxxxxxxx` may not fit in memory
							},
							VolumeMounts: []api.VolumeMount{
								{MountPath: "/test-empty-dir-mnt", Name: "test-empty-dir"},
							},
						},
					},
					Volumes: []api.Volume{
						{Name: "test-empty-dir", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
					},
				},
			},
		},
		{
			evictionPriority: 0, // This pod should never be evicted
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "normal-memory-usage-pod"},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image:           "gcr.io/google_containers/busybox:1.24",
							ImagePullPolicy: "Always",
							Name:            "normal-memory-usage-pod",
							Command: []string{
								"sh",
								"-c", //make one big (5 Gb) file
								"dd if=/dev/urandom of=largefile bs=5000000000 count=1; while true; do sleep 5; done",
							},
						},
					},
				},
			},
		},
	}
	evictionTestTimeout := 10 * time.Minute
	testCondition := "Disk Pressure due to Inodes"

	runEvictionTest(f, testCondition, podTestSpecs, evictionTestTimeout, hasInodePressure)
})

// Struct used by runEvictionTest that specifies the pod, and when that pod should be evicted, relative to other pods
type podTestSpec struct {
	evictionPriority int // 0 should never be evicted, 1 shouldn't evict before 2, etc.
	// If two are ranked at 1, either is permitted to fail before the other.
	// The test ends when all other than the 0 have been evicted
	pod api.Pod
}

// runEvictionTest sets up a testing environment given the provided nodes, and checks a few things:
//		It ensures that the desired testCondition is actually triggered.
//		It ensures that evictionPriority 0 pods are not evicted
//		It ensures that lower evictionPriority pods are always evicted before higher evictionPriority pods (2 evicted before 1, etc.)
//		It ensures that all lower evictionPriority pods are eventually evicted.
// runEvictionTest then cleans up the testing environment by deleting provided nodes, and ensures that testCondition no longer exists
func runEvictionTest(f *framework.Framework, testCondition string, podTestSpecs []podTestSpec,
	evictionTestTimeout time.Duration, hasPressureCondition func(*framework.Framework, string) (bool, error)) {

	Context(fmt.Sprintf("when we run containers that should cause %s", testCondition), func() {

		BeforeEach(func() {
			By("seting up pods to be used by tests")
			for _, spec := range podTestSpecs {
				By(fmt.Sprintf("creating pod with container: %s", spec.pod.Name))
				f.PodClient().Create(&spec.pod)
			}
		})

		It(fmt.Sprintf("should eventually see %s, and then evict all of the correct pods", testCondition), func() {
			Eventually(func() error {

				// Gather current information
				currentPods := getCurrentPods(f, podTestSpecs)
				hasPressure, err := hasPressureCondition(f, testCondition)
				framework.ExpectNoError(err, fmt.Sprintf("checking if we have %s", testCondition))

				if hasPressure {
					By("checking eviction ordering and ensuring important pods dont fail")
					done := true
					for i, priorityPodSpec := range podTestSpecs {
						priorityPod := currentPods[i]

						// Check eviction ordering.
						// Note: it is alright for a priority 1 and priority 2 pod (for example) to fail in the same round
						for j, lowPriorityPodSpec := range podTestSpecs {
							lowPriorityPod := currentPods[j]
							if priorityPodSpec.evictionPriority < lowPriorityPodSpec.evictionPriority && lowPriorityPod.Status.Phase == api.PodRunning {
								Expect(priorityPod.Status.Phase).NotTo(Equal(api.PodFailed),
									fmt.Sprintf("%s pod failed before %s pod", priorityPodSpec.pod.Name, lowPriorityPodSpec.pod.Name))
							}
						}

						// EvictionPriority 0 pods should not fail
						if priorityPodSpec.evictionPriority == 0 {
							Expect(priorityPod.Status.Phase).NotTo(Equal(api.PodFailed),
								fmt.Sprintf("%s pod failed (and shouldn't have failed)", priorityPodSpec.pod.Name))
						}

						// If a pod that is not evictionPriority 0 has not been evicted, we are not done
						if priorityPodSpec.evictionPriority != 0 && priorityPod.Status.Phase != api.PodFailed {
							done = false
						}
					}
					if done {
						return nil
					}
					return fmt.Errorf("pods that caused %s have not been evicted.", testCondition)

				}
				return fmt.Errorf("Condition: %s not encountered", testCondition)

			}, evictionTestTimeout, evictionPollInterval).Should(BeNil())
		})

		AfterEach(func() {
			By("making sure conditions eventually return to normal")
			Eventually(func() bool {
				hasPressure, err := hasPressureCondition(f, testCondition)
				framework.ExpectNoError(err, fmt.Sprintf("checking if we have %s", testCondition))
				return hasPressure
			}, evictionTestTimeout, evictionPollInterval).Should(BeFalse())

			By("making sure conditions do not return")
			Consistently(func() bool {
				hasPressure, err := hasPressureCondition(f, testCondition)
				framework.ExpectNoError(err, fmt.Sprintf("checking if we have %s", testCondition))
				return hasPressure
			}, postTestConditionMonitoringPeriod, evictionPollInterval).Should(BeFalse())

			By("making sure we can start a new pod after the test")
			podName := "admit-best-effort-pod"
			f.PodClient().Create(&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: podName,
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image: "gcr.io/google_containers/busybox:1.24",
							Name:  podName,
						},
					},
				},
			})
			if CurrentGinkgoTestDescription().Failed && framework.TestContext.DumpLogsOnFailure {
				logPodEvents(f)
				logNodeEvents(f)
			}
		})
	})
}

// Returns TRUE if the node has disk pressure due to inodes exists on the node, FALSE otherwise
func hasInodePressure(f *framework.Framework, testCondition string) (bool, error) {

	nodeList, err := f.ClientSet.Core().Nodes().List(api.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	if len(nodeList.Items) != 1 {
		return false, fmt.Errorf("expected 1 node, but see %d. List: %v", len(nodeList.Items), nodeList.Items)
	}

	_, pressure := api.GetNodeCondition(&nodeList.Items[0].Status, api.NodeDiskPressure)
	hasPressure := (*pressure).Status == api.ConditionTrue
	By(fmt.Sprintf("checking if pod has %s: %v", testCondition, hasPressure))

	// Additional Logging relating to Inodes
	summary, err := getNodeSummary()
	if err != nil {
		return false, err
	}
	framework.Logf("imageFsInfo.Inodes: %d, imageFsInfo.InodesFree: %d",
		*summary.Node.Runtime.ImageFs.Inodes, *summary.Node.Runtime.ImageFs.InodesFree)
	framework.Logf("rootFsInfo.Inodes: %d, rootFsInfo.InodesFree: %d",
		*summary.Node.Fs.Inodes, *summary.Node.Fs.InodesFree)
	for _, pod := range summary.Pods {
		framework.Logf("Pod: %s", pod.PodRef.Name)
		for _, container := range pod.Containers {
			if container.Rootfs != nil {
				framework.Logf("--- summary Container: %s inodeUsage: %d",
					container.Name, *container.Rootfs.InodesUsed)
			}
		}
		for _, volume := range pod.VolumeStats {
			if volume.FsStats.Inodes != nil && volume.FsStats.InodesUsed != nil {
				framework.Logf("--- summary Volume: %s inodeUsage: %d",
					volume.Name, *volume.FsStats.InodesUsed)
			}
		}
	}
	return hasPressure, nil
}

// Returns a list of pointers to the pods retrieved from the testing node
func getCurrentPods(f *framework.Framework, podTestSpecs []podTestSpec) (currentPods []*api.Pod) {
	By("getting current information on pods")

	for _, podSpec := range podTestSpecs {
		updatedPod, err := f.ClientSet.Core().Pods(f.Namespace.Name).Get(podSpec.pod.Name)
		framework.ExpectNoError(err, fmt.Sprintf("getting pod %s", podSpec.pod.Name))

		currentPods = append(currentPods, updatedPod)
		framework.Logf("fetching pod %s; phase= %v", updatedPod.Name, updatedPod.Status.Phase)
	}
	return
}
