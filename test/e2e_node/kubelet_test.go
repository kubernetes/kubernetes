/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"io/ioutil"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
)

var _ = Describe("Kubelet", func() {
	var cl *client.Client
	BeforeEach(func() {
		// Setup the apiserver client
		cl = client.NewOrDie(&client.Config{Host: *apiServerAddress})
	})

	Describe("pod scheduling", func() {
		Context("when scheduling a busybox command in a pod", func() {
			It("it should return succes", func() {
				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name:      "busybox",
						Namespace: api.NamespaceDefault,
					},
					Spec: api.PodSpec{
						// Force the Pod to schedule to the node without a scheduler running
						NodeName: *nodeName,
						// Don't restart the Pod since it is expected to exit
						RestartPolicy: api.RestartPolicyNever,
						Containers: []api.Container{
							{
								Image:   "gcr.io/google_containers/busybox",
								Name:    "busybox",
								Command: []string{"echo", "'Hello World'"},
							},
						},
					},
				}
				_, err := cl.Pods(api.NamespaceDefault).Create(pod)
				Expect(err).To(BeNil(), fmt.Sprintf("Error creating Pod %v", err))
			})

			It("it should print the output to logs", func() {
				errs := Retry(time.Minute, time.Second*4, func() error {
					rc, err := cl.Pods(api.NamespaceDefault).GetLogs("busybox", &api.PodLogOptions{}).Stream()
					if err != nil {
						return err
					}
					defer rc.Close()
					buf := new(bytes.Buffer)
					buf.ReadFrom(rc)
					if buf.String() != "'Hello World'\n" {
						return fmt.Errorf("Expected %s to match 'Hello World'", buf.String())
					}
					return nil
				})
				Expect(errs).To(BeEmpty(), fmt.Sprintf("Failed to get Logs"))
			})

			It("it should be possible to delete", func() {
				err := cl.Pods(api.NamespaceDefault).Delete("busybox", &api.DeleteOptions{})
				Expect(err).To(BeNil(), fmt.Sprintf("Error creating Pod %v", err))
			})
		})
	})

	Describe("metrics api", func() {
		statsPrefix := "stats-busybox-"
		podNames := []string{}
		podCount := 2
		for i := 0; i < podCount; i++ {
			podNames = append(podNames, fmt.Sprintf("%s%v", statsPrefix, i))
		}
		BeforeEach(func() {
			for _, podName := range podNames {
				createPod(cl, podName, []api.Container{
					{
						Image:   "gcr.io/google_containers/busybox",
						Command: []string{"sh", "-c", "echo 'Hello World' | tee ~/file | tee -a ~/file | tee /test-empty-dir | sleep 60"},
						Name:    podName + containerSuffix,
					},
				})
			}

			// Sleep long enough for cadvisor to see the pod and calculate all of its metrics
			time.Sleep(60 * time.Second)
		})

		Context("when querying /stats/summary", func() {
			It("it should report resource usage through the stats api", func() {
				resp, err := http.Get(*kubeletAddress + "/stats/summary")
				now := time.Now()
				Expect(err).To(BeNil(), fmt.Sprintf("Failed to get /stats/summary"))
				summary := stats.Summary{}
				contentsBytes, err := ioutil.ReadAll(resp.Body)
				Expect(err).To(BeNil(), fmt.Sprintf("Failed to read /stats/summary: %+v", resp))
				contents := string(contentsBytes)
				decoder := json.NewDecoder(strings.NewReader(contents))
				err = decoder.Decode(&summary)
				Expect(err).To(BeNil(), fmt.Sprintf("Failed to parse /stats/summary to go struct: %+v", resp))

				// Verify Misc Stats
				Expect(summary.Time.Time).To(BeTemporally("~", now, 20*time.Second))

				// Verify Node Stats are present
				Expect(summary.Node.NodeName).To(Equal(*nodeName))
				Expect(summary.Node.CPU.UsageCoreNanoSeconds).NotTo(BeZero())
				Expect(summary.Node.Memory.UsageBytes).NotTo(BeZero())
				Expect(summary.Node.Memory.WorkingSetBytes).NotTo(BeZero())
				Expect(summary.Node.Fs.UsedBytes).NotTo(BeZero())
				Expect(summary.Node.Fs.CapacityBytes).NotTo(BeZero())
				Expect(summary.Node.Fs.AvailableBytes).NotTo(BeZero())

				sysContainers := map[string]stats.ContainerStats{}
				sysContainersList := []string{}
				for _, container := range summary.Node.SystemContainers {
					sysContainers[container.Name] = container
					sysContainersList = append(sysContainersList, container.Name)
					Expect(container.CPU.UsageCoreNanoSeconds).NotTo(BeZero())
					// TODO: Test Network
					Expect(container.Memory.UsageBytes).NotTo(BeZero())
					Expect(container.Memory.WorkingSetBytes).NotTo(BeZero())
					Expect(container.Rootfs.CapacityBytes).NotTo(BeZero())
					Expect(container.Rootfs.AvailableBytes).NotTo(BeZero())
					Expect(container.Logs.CapacityBytes).NotTo(BeZero())
					Expect(container.Logs.AvailableBytes).NotTo(BeZero())
				}
				Expect(sysContainersList).To(ConsistOf("kubelet", "runtime"))

				// Verify Pods Stats are present
				podsList := []string{}
				for _, pod := range summary.Pods {
					if !strings.HasPrefix(pod.PodRef.Name, statsPrefix) {
						// Ignore pods created outside this test
						continue

					}
					// TODO: Test network

					podsList = append(podsList, pod.PodRef.Name)
					Expect(pod.Containers).To(HaveLen(1))
					container := pod.Containers[0]
					Expect(container.Name).To(Equal(pod.PodRef.Name + containerSuffix))
					Expect(container.CPU.UsageCoreNanoSeconds).NotTo(BeZero())
					Expect(container.Memory.UsageBytes).NotTo(BeZero())
					Expect(container.Memory.WorkingSetBytes).NotTo(BeZero())
					Expect(container.Rootfs.CapacityBytes).NotTo(BeZero())
					Expect(container.Rootfs.AvailableBytes).NotTo(BeZero())
					Expect(*container.Rootfs.UsedBytes).NotTo(BeZero(), contents)
					Expect(container.Logs.CapacityBytes).NotTo(BeZero())
					Expect(container.Logs.AvailableBytes).NotTo(BeZero())
					Expect(*container.Logs.UsedBytes).NotTo(BeZero(), contents)
				}
				Expect(podsList).To(ConsistOf(podNames))
			})
		})

		AfterEach(func() {
			for _, podName := range podNames {
				err := cl.Pods(api.NamespaceDefault).Delete(podName, &api.DeleteOptions{})
				Expect(err).To(BeNil(), fmt.Sprintf("Error deleting Pod %v", podName))
			}
		})
	})
})

const (
	containerSuffix = "-c"
)

func createPod(cl *client.Client, podName string, containers []api.Container) {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      podName,
			Namespace: api.NamespaceDefault,
		},
		Spec: api.PodSpec{
			// Force the Pod to schedule to the node without a scheduler running
			NodeName: *nodeName,
			// Don't restart the Pod since it is expected to exit
			RestartPolicy: api.RestartPolicyNever,
			Containers:    containers,
		},
	}
	_, err := cl.Pods(api.NamespaceDefault).Create(pod)
	Expect(err).To(BeNil(), fmt.Sprintf("Error creating Pod %v", err))
}
