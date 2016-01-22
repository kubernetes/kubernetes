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
								Image:           "gcr.io/google_containers/busybox",
								Name:            "busybox",
								Command:         []string{"echo", "'Hello World'"},
								ImagePullPolicy: api.PullIfNotPresent,
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
		pod1 := statsPrefix + "1"
		pod2 := statsPrefix + "2"
		containerSuffix := "-c"
		BeforeEach(func() {
			pod := &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:      pod1,
					Namespace: api.NamespaceDefault,
				},
				Spec: api.PodSpec{
					// Force the Pod to schedule to the node without a scheduler running
					NodeName: *nodeName,
					// Don't restart the Pod since it is expected to exit
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image:           "gcr.io/google_containers/pause:2.0",
							Name:            pod1 + containerSuffix,
							ImagePullPolicy: api.PullIfNotPresent,
						},
					},
				},
			}
			_, err := cl.Pods(api.NamespaceDefault).Create(pod)
			Expect(err).To(BeNil(), fmt.Sprintf("Error creating Pod %v", err))

			pod = &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:      pod2,
					Namespace: api.NamespaceDefault,
				},
				Spec: api.PodSpec{
					// Force the Pod to schedule to the node without a scheduler running
					NodeName: *nodeName,
					// Don't restart the Pod since it is expected to exit
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image:           "gcr.io/google_containers/pause:2.0",
							Name:            pod2 + containerSuffix,
							ImagePullPolicy: api.PullIfNotPresent,
						},
					},
				},
			}
			_, err = cl.Pods(api.NamespaceDefault).Create(pod)
			Expect(err).To(BeNil(), fmt.Sprintf("Error creating Pod %v", err))

			// Sleep long enough for cadvisor to see the pod and its metrics
			time.Sleep(30 * time.Second)
		})

		Context("when querying /stats/summary", func() {
			It("it should report resource usage through the stats api", func() {
				resp, err := http.Get(*kubeletAddress + "/stats/summary")
				now := time.Now()
				Expect(err).To(BeNil(), fmt.Sprintf("Failed to get /stats/summary"))
				summary := stats.Summary{}
				decoder := json.NewDecoder(resp.Body)
				err = decoder.Decode(&summary)
				Expect(err).To(BeNil(), fmt.Sprintf("Failed to parse /stats/summary to go struct: %+v", resp))

				// Verify Misc Stats
				Expect(summary.Time.Time).To(BeTemporally("~", now, 20*time.Second))

				// Verify Node Stats are present
				Expect(summary.Node.NodeName).To(Equal(*nodeName))
				Expect(summary.Node.CPU.UsageCoreSeconds).NotTo(BeZero())
				Expect(summary.Node.Memory.UsageBytes).NotTo(BeZero())
				Expect(summary.Node.Memory.WorkingSetBytes).NotTo(BeZero())
				// TODO: Test FS

				sysContainers := map[string]stats.ContainerStats{}
				sysContainersList := []string{}
				for _, container := range summary.Node.SystemContainers {
					sysContainers[container.Name] = container
					sysContainersList = append(sysContainersList, container.Name)
					Expect(container.CPU.UsageCoreSeconds).NotTo(BeZero())
					// TODO: Test Network
					// TODO: Test logs
					Expect(container.Memory.UsageBytes).NotTo(BeZero())
					Expect(container.Memory.WorkingSetBytes).NotTo(BeZero())
					// TODO: Test Rootfs
				}
				Expect(sysContainersList).To(ConsistOf("kubelet", "runtime"))

				// Verify Pods Stats are present
				pods := map[string]map[string]stats.ContainerStats{}
				podsList := []string{}
				for _, pod := range summary.Pods {
					if !strings.HasPrefix(pod.PodRef.Name, statsPrefix) {
						// Ignore pods created outside this test
						continue
					}
					// TODO: Test network

					m := map[string]stats.ContainerStats{}
					pods[pod.PodRef.Name] = m
					podsList = append(podsList, pod.PodRef.Name)
					for _, container := range pod.Containers {
						m[container.Name] = container
						Expect(container.CPU.UsageCoreSeconds).NotTo(BeZero())
						Expect(container.Memory.UsageBytes).NotTo(BeZero())
						Expect(container.Memory.WorkingSetBytes).NotTo(BeZero())
					}
				}

				Expect(podsList).To(ConsistOf(pod1, pod2))
				Expect(pods[pod1]).To(HaveKey(pod1 + containerSuffix))
				Expect(pods[pod2]).To(HaveKey(pod2 + containerSuffix))
			})
		})

		AfterEach(func() {
			err := cl.Pods(api.NamespaceDefault).Delete(pod1, &api.DeleteOptions{})
			Expect(err).To(BeNil(), fmt.Sprintf("Error deleting Pod"))
			err = cl.Pods(api.NamespaceDefault).Delete(pod2, &api.DeleteOptions{})
			Expect(err).To(BeNil(), fmt.Sprintf("Error deleting Pod"))
		})
	})
})
