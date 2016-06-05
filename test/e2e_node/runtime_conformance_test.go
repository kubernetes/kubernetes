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

type testCase struct {
	Name             string
	RestartPolicy    api.RestartPolicy
	Phase            api.PodPhase
	State            ContainerState
	RestartCountOper string
	RestartCount     int32
	Ready            bool
}

var _ = Describe("Container Runtime Conformance Test", func() {
	var cl *client.Client

	BeforeEach(func() {
		// Setup the apiserver client
		cl = client.NewOrDie(&restclient.Config{Host: *apiServerAddress})
	})

	Describe("container runtime conformance blackbox test", func() {
		namespace := "runtime-conformance"

		Context("when starting a container that exits", func() {
			It("it should run with the expected status [Conformance]", func() {
				restartCountVolumeName := "restart-count"
				restartCountVolumePath := "/restart-count"
				testContainer := api.Container{
					Image: ImageRegistry[busyBoxImage],
					VolumeMounts: []api.VolumeMount{
						{
							MountPath: restartCountVolumePath,
							Name:      restartCountVolumeName,
						},
					},
				}
				testVolumes := []api.Volume{
					{
						Name: restartCountVolumeName,
						VolumeSource: api.VolumeSource{
							HostPath: &api.HostPathVolumeSource{
								Path: os.TempDir(),
							},
						},
					},
				}
				testCases := []testCase{
					{"terminate-cmd-rpa", api.RestartPolicyAlways, api.PodRunning, ContainerStateRunning, "==", 2, true},
					{"terminate-cmd-rpof", api.RestartPolicyOnFailure, api.PodSucceeded, ContainerStateTerminated, "==", 1, false},
					{"terminate-cmd-rpn", api.RestartPolicyNever, api.PodFailed, ContainerStateTerminated, "==", 0, false},
				}
				for _, testCase := range testCases {
					tmpFile, err := ioutil.TempFile("", "restartCount")
					Expect(err).NotTo(HaveOccurred())
					defer os.Remove(tmpFile.Name())

					// It failed at the 1st run, then succeeded at 2nd run, then run forever
					cmdScripts := `
f=%s
count=$(echo 'hello' >> $f ; wc -l $f | awk {'print $1'})
if [ $count -eq 1 ]; then
	exit 1
fi
if [ $count -eq 2 ]; then
	exit 0
fi
while true; do sleep 1; done
`
					tmpCmd := fmt.Sprintf(cmdScripts, path.Join(restartCountVolumePath, path.Base(tmpFile.Name())))
					testContainer.Name = testCase.Name
					testContainer.Command = []string{"sh", "-c", tmpCmd}
					terminateContainer := ConformanceContainer{
						Container:     testContainer,
						Client:        cl,
						RestartPolicy: testCase.RestartPolicy,
						Volumes:       testVolumes,
						NodeName:      *nodeName,
						Namespace:     namespace,
					}
					Expect(terminateContainer.Create()).To(Succeed())
					defer terminateContainer.Delete()

					By("it should get the expected 'RestartCount'")
					Eventually(func() (int32, error) {
						status, err := terminateContainer.GetStatus()
						return status.RestartCount, err
					}, retryTimeout, pollInterval).Should(BeNumerically(testCase.RestartCountOper, testCase.RestartCount))

					By("it should get the expected 'Phase'")
					Eventually(terminateContainer.GetPhase, retryTimeout, pollInterval).Should(Equal(testCase.Phase))

					By("it should get the expected 'Ready' condition")
					Expect(terminateContainer.IsReady()).Should(Equal(testCase.Ready))

					status, err := terminateContainer.GetStatus()
					Expect(err).ShouldNot(HaveOccurred())

					By("it should get the expected 'State'")
					Expect(GetContainerState(status.State)).To(Equal(testCase.State))

					By("it should be possible to delete [Conformance]")
					Expect(terminateContainer.Delete()).To(Succeed())
					Eventually(terminateContainer.Present, retryTimeout, pollInterval).Should(BeFalse())
				}
			})
		})

		Context("when running a container with invalid image", func() {
			It("it should run with the expected status [Conformance]", func() {
				testContainer := api.Container{
					Image:   "foo.com/foo/foo",
					Command: []string{"false"},
				}
				testCase := testCase{"invalid-image-rpa", api.RestartPolicyAlways, api.PodPending, ContainerStateWaiting, "==", 0, false}
				testContainer.Name = testCase.Name
				invalidImageContainer := ConformanceContainer{
					Container:     testContainer,
					Client:        cl,
					RestartPolicy: testCase.RestartPolicy,
					NodeName:      *nodeName,
					Namespace:     namespace,
				}
				Expect(invalidImageContainer.Create()).To(Succeed())
				defer invalidImageContainer.Delete()

				Eventually(invalidImageContainer.GetPhase, retryTimeout, pollInterval).Should(Equal(testCase.Phase))
				Consistently(invalidImageContainer.GetPhase, consistentCheckTimeout, pollInterval).Should(Equal(testCase.Phase))

				status, err := invalidImageContainer.GetStatus()
				Expect(err).NotTo(HaveOccurred())

				By("it should get the expected 'RestartCount'")
				Expect(status.RestartCount).To(BeNumerically(testCase.RestartCountOper, testCase.RestartCount))

				By("it should get the expected 'Ready' status")
				Expect(status.Ready).To(Equal(testCase.Ready))

				By("it should get the expected 'State'")
				Expect(GetContainerState(status.State)).To(Equal(testCase.State))

				By("it should be possible to delete [Conformance]")
				Expect(invalidImageContainer.Delete()).To(Succeed())
				Eventually(invalidImageContainer.Present, retryTimeout, pollInterval).Should(BeFalse())
			})
		})
	})
})
