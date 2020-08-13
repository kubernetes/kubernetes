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

package e2enode

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	//TODO (dashpole): Once dynamic config is possible, test different values for maxPerPodContainer and maxContainers
	// Currently using default values for maxPerPodContainer and maxTotalContainers
	maxPerPodContainer = 1
	maxTotalContainers = -1

	garbageCollectDuration = 3 * time.Minute
	setupDuration          = 10 * time.Minute
	runtimePollInterval    = 10 * time.Second
)

type testPodSpec struct {
	podName string
	// containerPrefix must be unique for each pod, and cannot end in a number.
	// containerPrefix is used to identify which containers belong to which pod in the test.
	containerPrefix string
	// the number of times each container should restart
	restartCount int32
	// the number of containers in the test pod
	numContainers int
	// a function that returns the number of containers currently on the node (including dead containers).
	getContainerNames func() ([]string, error)
}

func (pod *testPodSpec) getContainerName(containerNumber int) string {
	return fmt.Sprintf("%s%d", pod.containerPrefix, containerNumber)
}

type testRun struct {
	// Name for logging purposes
	testName string
	// Pod specs for the test
	testPods []*testPodSpec
}

// GarbageCollect tests that the Kubelet conforms to the Kubelet Garbage Collection Policy, found here:
// http://kubernetes.io/docs/admin/garbage-collection/
var _ = framework.KubeDescribe("GarbageCollect [Serial][NodeFeature:GarbageCollect]", func() {
	f := framework.NewDefaultFramework("garbage-collect-test")
	containerNamePrefix := "gc-test-container-"
	podNamePrefix := "gc-test-pod-"

	// These suffixes are appended to pod and container names.
	// They differentiate pods from one another, and allow filtering
	// by names to identify which containers belong to which pods
	// They must be unique, and must not end in a number
	firstSuffix := "one-container-no-restarts"
	secondSuffix := "many-containers-many-restarts-one-pod"
	thirdSuffix := "many-containers-many-restarts-"
	tests := []testRun{
		{
			testName: "One Non-restarting Container",
			testPods: []*testPodSpec{
				{
					podName:         podNamePrefix + firstSuffix,
					containerPrefix: containerNamePrefix + firstSuffix,
					restartCount:    0,
					numContainers:   1,
				},
			},
		},
		{
			testName: "Many Restarting Containers",
			testPods: []*testPodSpec{
				{
					podName:         podNamePrefix + secondSuffix,
					containerPrefix: containerNamePrefix + secondSuffix,
					restartCount:    4,
					numContainers:   4,
				},
			},
		},
		{
			testName: "Many Pods with Many Restarting Containers",
			testPods: []*testPodSpec{
				{
					podName:         podNamePrefix + thirdSuffix + "one",
					containerPrefix: containerNamePrefix + thirdSuffix + "one",
					restartCount:    3,
					numContainers:   4,
				},
				{
					podName:         podNamePrefix + thirdSuffix + "two",
					containerPrefix: containerNamePrefix + thirdSuffix + "two",
					restartCount:    2,
					numContainers:   6,
				},
				{
					podName:         podNamePrefix + thirdSuffix + "three",
					containerPrefix: containerNamePrefix + thirdSuffix + "three",
					restartCount:    3,
					numContainers:   5,
				},
			},
		},
	}
	for _, test := range tests {
		containerGCTest(f, test)
	}
})

// Tests the following:
// 	pods are created, and all containers restart the specified number of times
// 	while containers are running, the number of copies of a single container does not exceed maxPerPodContainer
// 	while containers are running, the total number of containers does not exceed maxTotalContainers
// 	while containers are running, if not constrained by maxPerPodContainer or maxTotalContainers, keep an extra copy of each container
// 	once pods are killed, all containers are eventually cleaned up
func containerGCTest(f *framework.Framework, test testRun) {
	var runtime internalapi.RuntimeService
	ginkgo.BeforeEach(func() {
		var err error
		runtime, _, err = getCRIClient()
		framework.ExpectNoError(err)
	})
	for _, pod := range test.testPods {
		// Initialize the getContainerNames function to use CRI runtime client.
		pod.getContainerNames = func() ([]string, error) {
			relevantContainers := []string{}
			containers, err := runtime.ListContainers(&runtimeapi.ContainerFilter{
				LabelSelector: map[string]string{
					types.KubernetesPodNameLabel:      pod.podName,
					types.KubernetesPodNamespaceLabel: f.Namespace.Name,
				},
			})
			if err != nil {
				return relevantContainers, err
			}
			for _, container := range containers {
				relevantContainers = append(relevantContainers, container.Labels[types.KubernetesContainerNameLabel])
			}
			return relevantContainers, nil
		}
	}

	ginkgo.Context(fmt.Sprintf("Garbage Collection Test: %s", test.testName), func() {
		ginkgo.BeforeEach(func() {
			realPods := getPods(test.testPods)
			f.PodClient().CreateBatch(realPods)
			ginkgo.By("Making sure all containers restart the specified number of times")
			gomega.Eventually(func() error {
				for _, podSpec := range test.testPods {
					err := verifyPodRestartCount(f, podSpec.podName, podSpec.numContainers, podSpec.restartCount)
					if err != nil {
						return err
					}
				}
				return nil
			}, setupDuration, runtimePollInterval).Should(gomega.BeNil())
		})

		ginkgo.It(fmt.Sprintf("Should eventually garbage collect containers when we exceed the number of dead containers per container"), func() {
			totalContainers := 0
			for _, pod := range test.testPods {
				totalContainers += pod.numContainers*2 + 1
			}
			gomega.Eventually(func() error {
				total := 0
				for _, pod := range test.testPods {
					containerNames, err := pod.getContainerNames()
					if err != nil {
						return err
					}
					total += len(containerNames)
					// Check maxPerPodContainer for each container in the pod
					for i := 0; i < pod.numContainers; i++ {
						containerCount := 0
						for _, containerName := range containerNames {
							if containerName == pod.getContainerName(i) {
								containerCount++
							}
						}
						if containerCount > maxPerPodContainer+1 {
							return fmt.Errorf("expected number of copies of container: %s, to be <= maxPerPodContainer: %d; list of containers: %v",
								pod.getContainerName(i), maxPerPodContainer, containerNames)
						}
					}
				}
				//Check maxTotalContainers.  Currently, the default is -1, so this will never happen until we can configure maxTotalContainers
				if maxTotalContainers > 0 && totalContainers <= maxTotalContainers && total > maxTotalContainers {
					return fmt.Errorf("expected total number of containers: %v, to be <= maxTotalContainers: %v", total, maxTotalContainers)
				}
				return nil
			}, garbageCollectDuration, runtimePollInterval).Should(gomega.BeNil())

			if maxPerPodContainer >= 2 && maxTotalContainers < 0 { // make sure constraints wouldn't make us gc old containers
				ginkgo.By("Making sure the kubelet consistently keeps around an extra copy of each container.")
				gomega.Consistently(func() error {
					for _, pod := range test.testPods {
						containerNames, err := pod.getContainerNames()
						if err != nil {
							return err
						}
						for i := 0; i < pod.numContainers; i++ {
							containerCount := 0
							for _, containerName := range containerNames {
								if containerName == pod.getContainerName(i) {
									containerCount++
								}
							}
							if pod.restartCount > 0 && containerCount < maxPerPodContainer+1 {
								return fmt.Errorf("expected pod %v to have extra copies of old containers", pod.podName)
							}
						}
					}
					return nil
				}, garbageCollectDuration, runtimePollInterval).Should(gomega.BeNil())
			}
		})

		ginkgo.AfterEach(func() {
			for _, pod := range test.testPods {
				ginkgo.By(fmt.Sprintf("Deleting Pod %v", pod.podName))
				f.PodClient().DeleteSync(pod.podName, metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
			}

			ginkgo.By("Making sure all containers get cleaned up")
			gomega.Eventually(func() error {
				for _, pod := range test.testPods {
					containerNames, err := pod.getContainerNames()
					if err != nil {
						return err
					}
					if len(containerNames) > 0 {
						return fmt.Errorf("%v containers still remain", containerNames)
					}
				}
				return nil
			}, garbageCollectDuration, runtimePollInterval).Should(gomega.BeNil())

			if ginkgo.CurrentGinkgoTestDescription().Failed && framework.TestContext.DumpLogsOnFailure {
				logNodeEvents(f)
				logPodEvents(f)
			}
		})
	})
}

func getPods(specs []*testPodSpec) (pods []*v1.Pod) {
	for _, spec := range specs {
		ginkgo.By(fmt.Sprintf("Creating %v containers with restartCount: %v", spec.numContainers, spec.restartCount))
		containers := []v1.Container{}
		for i := 0; i < spec.numContainers; i++ {
			containers = append(containers, v1.Container{
				Image:   busyboxImage,
				Name:    spec.getContainerName(i),
				Command: getRestartingContainerCommand("/test-empty-dir-mnt", i, spec.restartCount, ""),
				VolumeMounts: []v1.VolumeMount{
					{MountPath: "/test-empty-dir-mnt", Name: "test-empty-dir"},
				},
			})
		}
		pods = append(pods, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: spec.podName},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyAlways,
				Containers:    containers,
				Volumes: []v1.Volume{
					{Name: "test-empty-dir", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}},
				},
			},
		})
	}
	return
}

func getRestartingContainerCommand(path string, containerNum int, restarts int32, loopingCommand string) []string {
	return []string{
		"sh",
		"-c",
		fmt.Sprintf(`
			f=%s/countfile%s
			count=$(echo 'hello' >> $f ; wc -l $f | awk {'print $1'})
			if [ $count -lt %d ]; then
				exit 0
			fi
			while true; do %s sleep 1; done`,
			path, strconv.Itoa(containerNum), restarts+1, loopingCommand),
	}
}

func verifyPodRestartCount(f *framework.Framework, podName string, expectedNumContainers int, expectedRestartCount int32) error {
	updatedPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), podName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	if len(updatedPod.Status.ContainerStatuses) != expectedNumContainers {
		return fmt.Errorf("expected pod %s to have %d containers, actual: %d",
			updatedPod.Name, expectedNumContainers, len(updatedPod.Status.ContainerStatuses))
	}
	for _, containerStatus := range updatedPod.Status.ContainerStatuses {
		if containerStatus.RestartCount != expectedRestartCount {
			return fmt.Errorf("pod %s had container with restartcount %d.  Should have been at least %d",
				updatedPod.Name, containerStatus.RestartCount, expectedRestartCount)
		}
	}
	return nil
}
