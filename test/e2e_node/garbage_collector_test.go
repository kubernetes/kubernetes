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

	"k8s.io/kubernetes/pkg/api"
	docker "k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	defaultDockerEndpoint = "unix:///var/run/docker.sock"

	//TODO (dashpole): Once dynamic config is possible, test different values for maxPerPodContainer and maxContainers
	// Currently using default values for maxPerPodContainer and maxTotalContainers
	maxPerPodContainer = 1
	maxTotalContainers = -1

	defaultRuntimeRequestTimeoutDuration = 1 * time.Minute
	garbageCollectDuration               = 2 * time.Minute
	setupDuration                        = 10 * time.Minute
	runtimePollInterval                  = 10 * time.Second
	deleteTimeout                        = 4 * time.Minute
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
var _ = framework.KubeDescribe("GarbageCollect [Serial]", func() {
	f := framework.NewDefaultFramework("garbage-collect-test")
	containerNamePrefix := "gc-test-container-"
	podNamePrefix := "gc-test-pod-"

	// These suffixes are appended to pod and container names.
	// They differentiate pods from one another, and allow filtering
	// by names to identify which containers belong to which pods
	// They must be unique, and must not end in a number
	first_suffix := "one-container-no-restarts"
	second_suffix := "many-containers-many-restarts-one-pod"
	third_suffix := "many-containers-many-restarts-"
	tests := []testRun{
		{
			testName: "One Non-restarting Container",
			testPods: []*testPodSpec{
				{
					podName:         podNamePrefix + first_suffix,
					containerPrefix: containerNamePrefix + first_suffix,
					restartCount:    0,
					numContainers:   1,
				},
			},
		},
		{
			testName: "Many Restarting Containers",
			testPods: []*testPodSpec{
				{
					podName:         podNamePrefix + second_suffix,
					containerPrefix: containerNamePrefix + second_suffix,
					restartCount:    4,
					numContainers:   4,
				},
			},
		},
		{
			testName: "Many Pods with Many Restarting Containers",
			testPods: []*testPodSpec{
				{
					podName:         podNamePrefix + third_suffix + "one",
					containerPrefix: containerNamePrefix + third_suffix + "one",
					restartCount:    3,
					numContainers:   4,
				},
				{
					podName:         podNamePrefix + third_suffix + "two",
					containerPrefix: containerNamePrefix + third_suffix + "two",
					restartCount:    2,
					numContainers:   6,
				},
				{
					podName:         podNamePrefix + third_suffix + "three",
					containerPrefix: containerNamePrefix + third_suffix + "three",
					restartCount:    3,
					numContainers:   5,
				},
			},
		},
	}
	for _, test := range tests {
		// TODO (dashpole): Once the Container Runtime Interface (CRI) is complete, generalize run on other runtimes (other than docker)
		dockerContainerGCTest(f, test)
	}
})

// Tests the following:
// 	pods are created, and all containers restart the specified number of times
// 	while contianers are running, the number of copies of a single container does not exceed maxPerPodContainer
// 	while containers are running, the total number of containers does not exceed maxTotalContainers
// 	while containers are running, if not constrained by maxPerPodContainer or maxTotalContainers, keep an extra copy of each container
// 	once pods are killed, all containers are eventually cleaned up
func containerGCTest(f *framework.Framework, test testRun) {
	Context(fmt.Sprintf("Garbage Collection Test: %s", test.testName), func() {
		BeforeEach(func() {
			realPods := getPods(test.testPods)
			f.PodClient().CreateBatch(realPods)
			By("Making sure all containers restart the specified number of times")
			Eventually(func() error {
				for _, podSpec := range test.testPods {
					updatedPod, err := f.ClientSet.Core().Pods(f.Namespace.Name).Get(podSpec.podName)
					if err != nil {
						return err
					}
					if len(updatedPod.Status.ContainerStatuses) != podSpec.numContainers {
						return fmt.Errorf("expected pod %s to have %d containers, actual: %d",
							updatedPod.Name, podSpec.numContainers, len(updatedPod.Status.ContainerStatuses))
					}
					for _, containerStatus := range updatedPod.Status.ContainerStatuses {
						if containerStatus.RestartCount != podSpec.restartCount {
							return fmt.Errorf("pod %s had container with restartcount %d.  Should have been at least %d",
								updatedPod.Name, containerStatus.RestartCount, podSpec.restartCount)
						}
					}
				}
				return nil
			}, setupDuration, runtimePollInterval).Should(BeNil())
		})

		It(fmt.Sprintf("Should eventually garbage collect containers when we exceed the number of dead containers per container"), func() {
			totalContainers := 0
			for _, pod := range test.testPods {
				totalContainers += pod.numContainers*2 + 1
			}
			Eventually(func() error {
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
							if strings.Contains(containerName, pod.getContainerName(i)) {
								containerCount += 1
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
			}, garbageCollectDuration, runtimePollInterval).Should(BeNil())

			if maxPerPodContainer >= 2 && maxTotalContainers < 0 { // make sure constraints wouldn't make us gc old containers
				By("Making sure the kubelet consistently keeps around an extra copy of each container.")
				Consistently(func() error {
					for _, pod := range test.testPods {
						containerNames, err := pod.getContainerNames()
						if err != nil {
							return err
						}
						for i := 0; i < pod.numContainers; i++ {
							containerCount := 0
							for _, containerName := range containerNames {
								if strings.Contains(containerName, pod.getContainerName(i)) {
									containerCount += 1
								}
							}
							if pod.restartCount > 0 && containerCount < maxPerPodContainer+1 {
								return fmt.Errorf("expected pod %v to have extra copies of old containers", pod.podName)
							}
						}
					}
					return nil
				}, garbageCollectDuration, runtimePollInterval).Should(BeNil())
			}
		})

		AfterEach(func() {
			for _, pod := range test.testPods {
				By(fmt.Sprintf("Deleting Pod %v", pod.podName))
				f.PodClient().DeleteSync(pod.podName, &api.DeleteOptions{}, defaultRuntimeRequestTimeoutDuration)
			}

			By("Making sure all containers get cleaned up")
			Eventually(func() error {
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
			}, garbageCollectDuration, runtimePollInterval).Should(BeNil())

			if CurrentGinkgoTestDescription().Failed && framework.TestContext.DumpLogsOnFailure {
				logNodeEvents(f)
				logPodEvents(f)
			}
		})
	})
}

// Runs containerGCTest using the docker runtime.
func dockerContainerGCTest(f *framework.Framework, test testRun) {
	var runtime docker.DockerInterface
	BeforeEach(func() {
		runtime = docker.ConnectToDockerOrDie(defaultDockerEndpoint, defaultRuntimeRequestTimeoutDuration)
	})
	for _, pod := range test.testPods {
		// Initialize the getContainerNames function to use the dockertools api
		thisPrefix := pod.containerPrefix
		pod.getContainerNames = func() ([]string, error) {
			relevantContainers := []string{}
			dockerContainers, err := docker.GetKubeletDockerContainers(runtime, true)
			if err != nil {
				return relevantContainers, err
			}
			for _, container := range dockerContainers {
				// only look for containers from this testspec
				if strings.Contains(container.Names[0], thisPrefix) {
					relevantContainers = append(relevantContainers, container.Names[0])
				}
			}
			return relevantContainers, nil
		}
	}
	containerGCTest(f, test)
}

func getPods(specs []*testPodSpec) (pods []*api.Pod) {
	for _, spec := range specs {
		By(fmt.Sprintf("Creating %v containers with restartCount: %v", spec.numContainers, spec.restartCount))
		containers := []api.Container{}
		for i := 0; i < spec.numContainers; i++ {
			containers = append(containers, api.Container{
				Image: "gcr.io/google_containers/busybox:1.24",
				Name:  spec.getContainerName(i),
				Command: []string{
					"sh",
					"-c",
					fmt.Sprintf(`
						f=/test-empty-dir-mnt/countfile%d
						count=$(echo 'hello' >> $f ; wc -l $f | awk {'print $1'})
						if [ $count -lt %d ]; then
							exit 0
						fi
						while true; do sleep 1; done
					`, i, spec.restartCount+1),
				},
				VolumeMounts: []api.VolumeMount{
					{MountPath: "/test-empty-dir-mnt", Name: "test-empty-dir"},
				},
			})
		}
		pods = append(pods, &api.Pod{
			ObjectMeta: api.ObjectMeta{Name: spec.podName},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				Containers:    containers,
				Volumes: []api.Volume{
					{Name: "test-empty-dir", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				},
			},
		})
	}
	return
}
