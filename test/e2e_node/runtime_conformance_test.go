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

var _ = Describe("Container runtime Conformance Test", func() {
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
				testStatuses := []testStatus{
					{"terminate-cmd-rpa", api.RestartPolicyAlways, api.PodRunning, ContainerStateRunning, "==", 2, true},
					{"terminate-cmd-rpof", api.RestartPolicyOnFailure, api.PodSucceeded, ContainerStateTerminated, "==", 1, false},
					{"terminate-cmd-rpn", api.RestartPolicyNever, api.PodFailed, ContainerStateTerminated, "==", 0, false},
				}

				for _, testStatus := range testStatuses {
					tmpFile, err := ioutil.TempFile("", "restartCount")
					Expect(err).NotTo(HaveOccurred())
					defer os.Remove(tmpFile.Name())

					// It fails in the first three runs and succeeds after that.
					cmdScripts := `
f=%s
count=$(echo 'hello' >> $f ; wc -l $f | awk {'print $1'})
if [ $count -eq 1 ]; then
	exit 1
elif [ $count -eq 2 ]; then
	exit 0
else
	while true; do sleep 1; done
	exit 0
fi`
					tmpCmd := fmt.Sprintf(cmdScripts, path.Base(tmpFile.Name()))
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

					By("it should get the expected 'Ready' condition")
					Eventually(func() bool {
						isReady, _ := terminateContainer.IsReady()
						return isReady
					}, retryTimeout, pollInterval).Should(Equal(testStatus.Ready))

					var phase api.PodPhase
					status, phase, err = terminateContainer.GetStatus()
					Expect(err).NotTo(HaveOccurred())

					By("it should get the expected 'State'")
					Expect(GetContainerState(status.State) & testStatus.State).NotTo(Equal(0))

					By("it should get the expected 'Phase'")
					Expect(phase).To(Equal(testStatus.Phase))

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
