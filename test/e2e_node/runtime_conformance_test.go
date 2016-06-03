/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"path"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	consistentCheckTimeout = time.Second * 10
	retryTimeout           = time.Minute * 5
	pollInterval           = time.Second * 5
)

type testStatus struct {
	Name             string
	RestartPolicy    api.RestartPolicy
	Phase            api.PodPhase
	State            ContainerState
	RestartCountOper string
	RestartCount     int32
	Ready            bool
}

var _ = Describe("[FLAKY] Container runtime Conformance Test", func() {
	var cl *client.Client

	BeforeEach(func() {
		// Setup the apiserver client
		cl = client.NewOrDie(&restclient.Config{Host: *apiServerAddress})
	})

	Describe("container runtime conformance blackbox test", func() {
		var testCContainers []ConformanceContainer
		namespace := "runtime-conformance"

		BeforeEach(func() {
			testCContainers = []ConformanceContainer{}
		})

		Context("when start a container that exits successfully", func() {
			It("it should run with the expected status [Conformance]", func() {
				testContainer := api.Container{
					Image: ImageRegistry[busyBoxImage],
					VolumeMounts: []api.VolumeMount{
						{
							MountPath: "/restart-count",
							Name:      "restart-count",
						},
					},
					ImagePullPolicy: api.PullIfNotPresent,
				}
				testVolumes := []api.Volume{
					{
						Name: "restart-count",
						VolumeSource: api.VolumeSource{
							HostPath: &api.HostPathVolumeSource{
								Path: os.TempDir(),
							},
						},
					},
				}
				testCount := int32(3)
				testStatuses := []testStatus{
					{"terminate-cmd-rpa", api.RestartPolicyAlways, api.PodRunning, ContainerStateWaiting | ContainerStateRunning | ContainerStateTerminated, ">", testCount, false},
					{"terminate-cmd-rpof", api.RestartPolicyOnFailure, api.PodSucceeded, ContainerStateTerminated, "==", testCount, false},
					{"terminate-cmd-rpn", api.RestartPolicyNever, api.PodSucceeded, ContainerStateTerminated, "==", 0, false},
				}

				for _, testStatus := range testStatuses {
					tmpFile, err := ioutil.TempFile("", "restartCount")
					Expect(err).NotTo(HaveOccurred())
					defer os.Remove(tmpFile.Name())

					// It fails in the first three runs and succeeds after that.
					tmpCmd := fmt.Sprintf("echo 'hello' >> /restart-count/%s ; test $(wc -l /restart-count/%s| awk {'print $1'}) -ge %d", path.Base(tmpFile.Name()), path.Base(tmpFile.Name()), testCount+1)
					testContainer.Name = testStatus.Name
					testContainer.Command = []string{"sh", "-c", tmpCmd}
					terminateContainer := ConformanceContainer{
						Container:     testContainer,
						Client:        cl,
						RestartPolicy: testStatus.RestartPolicy,
						Volumes:       testVolumes,
						NodeName:      *nodeName,
						Namespace:     namespace,
					}
					err = terminateContainer.Create()
					Expect(err).NotTo(HaveOccurred())
					testCContainers = append(testCContainers, terminateContainer)

					Eventually(func() api.PodPhase {
						_, phase, _ := terminateContainer.GetStatus()
						return phase
					}, retryTimeout, pollInterval).ShouldNot(Equal(api.PodPending))

					var status api.ContainerStatus
					By("it should get the expected 'RestartCount'")
					Eventually(func() int32 {
						status, _, _ = terminateContainer.GetStatus()
						return status.RestartCount
					}, retryTimeout, pollInterval).Should(BeNumerically(testStatus.RestartCountOper, testStatus.RestartCount))

					By("it should get the expected 'Ready' status")
					Expect(status.Ready).To(Equal(testStatus.Ready))

					By("it should get the expected 'State'")
					Expect(GetContainerState(status.State) & testStatus.State).NotTo(Equal(0))

					By("it should be possible to delete [Conformance]")
					err = terminateContainer.Delete()
					Expect(err).NotTo(HaveOccurred())
					Eventually(func() bool {
						isPresent, err := terminateContainer.Present()
						return err == nil && !isPresent
					}, retryTimeout, pollInterval).Should(BeTrue())
				}
			})
		})

		Context("when start a container that keeps running", func() {
			It("it should run with the expected status [Conformance]", func() {
				testContainer := api.Container{
					Image:           ImageRegistry[busyBoxImage],
					Command:         []string{"sh", "-c", "while true; do echo hello; sleep 1; done"},
					ImagePullPolicy: api.PullIfNotPresent,
				}
				testStatuses := []testStatus{
					{"loop-cmd-rpa", api.RestartPolicyAlways, api.PodRunning, ContainerStateRunning, "==", 0, true},
					{"loop-cmd-rpof", api.RestartPolicyOnFailure, api.PodRunning, ContainerStateRunning, "==", 0, true},
					{"loop-cmd-rpn", api.RestartPolicyNever, api.PodRunning, ContainerStateRunning, "==", 0, true},
				}
				for _, testStatus := range testStatuses {
					testContainer.Name = testStatus.Name
					runningContainer := ConformanceContainer{
						Container:     testContainer,
						Client:        cl,
						RestartPolicy: testStatus.RestartPolicy,
						NodeName:      *nodeName,
						Namespace:     namespace,
					}
					err := runningContainer.Create()
					Expect(err).NotTo(HaveOccurred())
					testCContainers = append(testCContainers, runningContainer)

					Eventually(func() api.PodPhase {
						_, phase, _ := runningContainer.GetStatus()
						return phase
					}, retryTimeout, pollInterval).Should(Equal(api.PodRunning))

					var status api.ContainerStatus
					var phase api.PodPhase
					Consistently(func() api.PodPhase {
						status, phase, err = runningContainer.GetStatus()
						return phase
					}, consistentCheckTimeout, pollInterval).Should(Equal(testStatus.Phase))
					Expect(err).NotTo(HaveOccurred())

					By("it should get the expected 'RestartCount'")
					Expect(status.RestartCount).To(BeNumerically(testStatus.RestartCountOper, testStatus.RestartCount))

					By("it should get the expected 'Ready' status")
					Expect(status.Ready).To(Equal(testStatus.Ready))

					By("it should get the expected 'State'")
					Expect(GetContainerState(status.State) & testStatus.State).NotTo(Equal(0))

					By("it should be possible to delete [Conformance]")
					err = runningContainer.Delete()
					Expect(err).NotTo(HaveOccurred())
					Eventually(func() bool {
						isPresent, err := runningContainer.Present()
						return err == nil && !isPresent
					}, retryTimeout, pollInterval).Should(BeTrue())
				}
			})
		})

		Context("when start a container that exits failure", func() {
			It("it should run with the expected status [Conformance]", func() {
				testContainer := api.Container{
					Image:           ImageRegistry[busyBoxImage],
					Command:         []string{"false"},
					ImagePullPolicy: api.PullIfNotPresent,
				}
				testStatuses := []testStatus{
					{"fail-cmd-rpa", api.RestartPolicyAlways, api.PodRunning, ContainerStateWaiting | ContainerStateRunning | ContainerStateTerminated, ">", 0, false},
					{"fail-cmd-rpof", api.RestartPolicyOnFailure, api.PodRunning, ContainerStateTerminated, ">", 0, false},
					{"fail-cmd-rpn", api.RestartPolicyNever, api.PodFailed, ContainerStateTerminated, "==", 0, false},
				}
				for _, testStatus := range testStatuses {
					testContainer.Name = testStatus.Name
					failureContainer := ConformanceContainer{
						Container:     testContainer,
						Client:        cl,
						RestartPolicy: testStatus.RestartPolicy,
						NodeName:      *nodeName,
						Namespace:     namespace,
					}
					err := failureContainer.Create()
					Expect(err).NotTo(HaveOccurred())
					testCContainers = append(testCContainers, failureContainer)

					Eventually(func() api.PodPhase {
						_, phase, _ := failureContainer.GetStatus()
						return phase
					}, retryTimeout, pollInterval).ShouldNot(Equal(api.PodPending))

					var status api.ContainerStatus
					By("it should get the expected 'RestartCount'")
					Eventually(func() int32 {
						status, _, _ = failureContainer.GetStatus()
						return status.RestartCount
					}, retryTimeout, pollInterval).Should(BeNumerically(testStatus.RestartCountOper, testStatus.RestartCount))

					By("it should get the expected 'Ready' status")
					Expect(status.Ready).To(Equal(testStatus.Ready))

					By("it should get the expected 'State'")
					Expect(GetContainerState(status.State) & testStatus.State).NotTo(Equal(0))

					By("it should be possible to delete [Conformance]")
					err = failureContainer.Delete()
					Expect(err).NotTo(HaveOccurred())
					Eventually(func() bool {
						isPresent, err := failureContainer.Present()
						return err == nil && !isPresent
					}, retryTimeout, pollInterval).Should(BeTrue())
				}
			})
		})

		Context("when running a container with invalid image", func() {
			It("it should run with the expected status [Conformance]", func() {
				testContainer := api.Container{
					Image:           "foo.com/foo/foo",
					Command:         []string{"false"},
					ImagePullPolicy: api.PullIfNotPresent,
				}
				testStatus := testStatus{"invalid-image-rpa", api.RestartPolicyAlways, api.PodPending, ContainerStateWaiting, "==", 0, false}
				testContainer.Name = testStatus.Name
				invalidImageContainer := ConformanceContainer{
					Container:     testContainer,
					Client:        cl,
					RestartPolicy: testStatus.RestartPolicy,
					NodeName:      *nodeName,
					Namespace:     namespace,
				}
				err := invalidImageContainer.Create()
				Expect(err).NotTo(HaveOccurred())
				testCContainers = append(testCContainers, invalidImageContainer)

				var status api.ContainerStatus
				var phase api.PodPhase

				Consistently(func() api.PodPhase {
					if status, phase, err = invalidImageContainer.GetStatus(); err != nil {
						return api.PodPending
					} else {
						return phase
					}
				}, consistentCheckTimeout, pollInterval).Should(Equal(testStatus.Phase))
				Expect(err).NotTo(HaveOccurred())

				By("it should get the expected 'RestartCount'")
				Expect(status.RestartCount).To(BeNumerically(testStatus.RestartCountOper, testStatus.RestartCount))

				By("it should get the expected 'Ready' status")
				Expect(status.Ready).To(Equal(testStatus.Ready))

				By("it should get the expected 'State'")
				Expect(GetContainerState(status.State) & testStatus.State).NotTo(Equal(0))

				By("it should be possible to delete [Conformance]")
				err = invalidImageContainer.Delete()
				Expect(err).NotTo(HaveOccurred())
				Eventually(func() bool {
					isPresent, err := invalidImageContainer.Present()
					return err == nil && !isPresent
				}, retryTimeout, pollInterval).Should(BeTrue())
			})
		})

		AfterEach(func() {
			for _, cc := range testCContainers {
				cc.Delete()
			}
		})
	})
})
