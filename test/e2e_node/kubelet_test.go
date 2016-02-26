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
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"

	"github.com/davecgh/go-spew/spew"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
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
				Eventually(func() string {
					rc, err := cl.Pods(api.NamespaceDefault).GetLogs("busybox", &api.PodLogOptions{}).Stream()
					if err != nil {
						return ""
					}
					defer rc.Close()
					buf := new(bytes.Buffer)
					buf.ReadFrom(rc)
					return buf.String()
				}, time.Second*30, time.Second*4).Should(Equal("'Hello World'\n"))
			})

			It("it should be possible to delete", func() {
				err := cl.Pods(api.NamespaceDefault).Delete("busybox", &api.DeleteOptions{})
				Expect(err).To(BeNil(), fmt.Sprintf("Error creating Pod %v", err))
			})
		})

		// TODO: Enable this when issues are resolved.  Tracked in #21320
		//		Context("when scheduling a read only busybox container", func() {
		//			It("it should return success", func() {
		//				pod := &api.Pod{
		//					ObjectMeta: api.ObjectMeta{
		//						Name:      "busybox",
		//						Namespace: api.NamespaceDefault,
		//					},
		//					Spec: api.PodSpec{
		//						// Force the Pod to schedule to the node without a scheduler running
		//						NodeName: *nodeName,
		//						// Don't restart the Pod since it is expected to exit
		//						RestartPolicy: api.RestartPolicyNever,
		//						Containers: []api.Container{
		//							{
		//								Image:   "gcr.io/google_containers/busybox",
		//								Name:    "busybox",
		//								Command: []string{"sh", "-c", "echo test > /file"},
		//								SecurityContext: &api.SecurityContext{
		//									ReadOnlyRootFilesystem: &isReadOnly,
		//								},
		//							},
		//						},
		//					},
		//				}
		//				_, err := cl.Pods(api.NamespaceDefault).Create(pod)
		//				Expect(err).To(BeNil(), fmt.Sprintf("Error creating Pod %v", err))
		//			})
		//
		//			It("it should not write to the root filesystem", func() {
		//				Eventually(func() string {
		//					rc, err := cl.Pods(api.NamespaceDefault).GetLogs("busybox", &api.PodLogOptions{}).Stream()
		//					if err != nil {
		//						return ""
		//					}
		//					defer rc.Close()
		//					buf := new(bytes.Buffer)
		//					buf.ReadFrom(rc)
		//					return buf.String()
		//				}, time.Second*30, time.Second*4).Should(Equal("sh: can't create /file: Read-only file system"))
		//			})
		//
		//			It("it should be possible to delete", func() {
		//				err := cl.Pods(api.NamespaceDefault).Delete("busybox", &api.DeleteOptions{})
		//				Expect(err).To(BeNil(), fmt.Sprintf("Error creating Pod %v", err))
		//			})
		//		})
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
						Command: []string{"sh", "-c", "echo 'Hello World' | tee ~/file | tee /test-empty-dir-mnt | sleep 60"},
						Name:    podName + containerSuffix,
						VolumeMounts: []api.VolumeMount{
							{MountPath: "/test-empty-dir-mnt", Name: "test-empty-dir"},
						},
					},
				}, []api.Volume{
					// TODO: Test secret volumes
					// TODO: Test hostpath volumes
					{Name: "test-empty-dir", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				})
			}

			// Sleep long enough for cadvisor to see the pod and calculate all of its metrics
			time.Sleep(60 * time.Second)
		})

		Context("when querying /stats/summary", func() {
			It("it should report resource usage through the stats api", func() {
				By("Returning stats summary")
				resp, err := http.Get(*kubeletAddress + "/stats/summary")
				Expect(err).To(BeNil(), fmt.Sprintf("Failed to get /stats/summary"))
				summary := stats.Summary{}
				contentsBytes, err := ioutil.ReadAll(resp.Body)
				Expect(err).To(BeNil(), fmt.Sprintf("Failed to read /stats/summary: %+v", resp))
				contents := string(contentsBytes)
				decoder := json.NewDecoder(strings.NewReader(contents))
				err = decoder.Decode(&summary)
				Expect(err).To(BeNil(), fmt.Sprintf("Failed to parse /stats/summary to go struct: %+v", resp))

				By("Having resources for node")
				Expect(summary.Node.NodeName).To(Equal(*nodeName))
				Expect(summary.Node.CPU.UsageCoreNanoSeconds).NotTo(BeNil())
				Expect(*summary.Node.CPU.UsageCoreNanoSeconds).NotTo(BeZero())

				Expect(summary.Node.Memory.UsageBytes).NotTo(BeNil())
				Expect(*summary.Node.Memory.UsageBytes).NotTo(BeZero())

				Expect(summary.Node.Memory.WorkingSetBytes).NotTo(BeNil())
				Expect(*summary.Node.Memory.WorkingSetBytes).NotTo(BeZero())

				Expect(summary.Node.Fs.AvailableBytes).NotTo(BeNil())
				Expect(*summary.Node.Fs.AvailableBytes).NotTo(BeZero())
				Expect(summary.Node.Fs.CapacityBytes).NotTo(BeNil())
				Expect(*summary.Node.Fs.CapacityBytes).NotTo(BeZero())
				Expect(summary.Node.Fs.UsedBytes).NotTo(BeNil())
				Expect(*summary.Node.Fs.UsedBytes).NotTo(BeZero())

				By("Having resources for kubelet and runtime system containers")
				sysContainers := map[string]stats.ContainerStats{}
				sysContainersList := []string{}
				for _, container := range summary.Node.SystemContainers {
					sysContainers[container.Name] = container
					sysContainersList = append(sysContainersList, container.Name)
					ExpectContainerStatsNotEmpty(&container)
				}
				Expect(sysContainersList).To(ConsistOf("kubelet", "runtime"))

				// Verify Pods Stats are present
				podsList := []string{}
				By("Having resources for pods")
				for _, pod := range summary.Pods {
					if !strings.HasPrefix(pod.PodRef.Name, statsPrefix) {
						// Ignore pods created outside this test
						continue
					}

					podsList = append(podsList, pod.PodRef.Name)

					Expect(pod.Containers).To(HaveLen(1))
					container := pod.Containers[0]
					Expect(container.Name).To(Equal(pod.PodRef.Name + containerSuffix))

					ExpectContainerStatsNotEmpty(&container)

					// emptydir volume
					volumeNames := []string{}
					for _, vs := range pod.VolumeStats {
						Expect(vs.CapacityBytes).NotTo(BeZero())
						Expect(vs.AvailableBytes).NotTo(BeZero())
						Expect(vs.UsedBytes).NotTo(BeZero())
						volumeNames = append(volumeNames, vs.Name)
					}
					Expect(volumeNames).To(ConsistOf("test-empty-dir"))

					// fs usage (not for system containers)
					Expect(container.Rootfs).NotTo(BeNil(), spew.Sdump(container))
					Expect(container.Rootfs.AvailableBytes).NotTo(BeNil(), spew.Sdump(container))
					Expect(*container.Rootfs.AvailableBytes).NotTo(BeZero(), spew.Sdump(container))
					Expect(container.Rootfs.CapacityBytes).NotTo(BeNil(), spew.Sdump(container))
					Expect(*container.Rootfs.CapacityBytes).NotTo(BeZero(), spew.Sdump(container))
					Expect(container.Rootfs.UsedBytes).NotTo(BeNil(), spew.Sdump(container))
					Expect(*container.Rootfs.UsedBytes).NotTo(BeZero(), spew.Sdump(container))
					Expect(container.Logs).NotTo(BeNil(), spew.Sdump(container))
					Expect(container.Logs.AvailableBytes).NotTo(BeNil(), spew.Sdump(container))
					Expect(*container.Logs.AvailableBytes).NotTo(BeZero(), spew.Sdump(container))
					Expect(container.Logs.CapacityBytes).NotTo(BeNil(), spew.Sdump(container))
					Expect(*container.Logs.CapacityBytes).NotTo(BeZero(), spew.Sdump(container))
					Expect(container.Logs.UsedBytes).NotTo(BeNil(), spew.Sdump(container))
					Expect(*container.Logs.UsedBytes).NotTo(BeZero(), spew.Sdump(container))

				}
				Expect(podsList).To(ConsistOf(podNames), spew.Sdump(summary))
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

func ExpectContainerStatsNotEmpty(container *stats.ContainerStats) {
	// TODO: Test Network

	Expect(container.CPU).NotTo(BeNil(), spew.Sdump(container))
	Expect(container.CPU.UsageCoreNanoSeconds).NotTo(BeNil(), spew.Sdump(container))
	Expect(*container.CPU.UsageCoreNanoSeconds).NotTo(BeZero(), spew.Sdump(container))

	Expect(container.Memory).NotTo(BeNil(), spew.Sdump(container))
	Expect(container.Memory.UsageBytes).NotTo(BeNil(), spew.Sdump(container))
	Expect(*container.Memory.UsageBytes).NotTo(BeZero(), spew.Sdump(container))
	Expect(container.Memory.WorkingSetBytes).NotTo(BeNil(), spew.Sdump(container))
	Expect(*container.Memory.WorkingSetBytes).NotTo(BeZero(), spew.Sdump(container))
}

const (
	containerSuffix = "-c"
)

func createPod(cl *client.Client, podName string, containers []api.Container, volumes []api.Volume) {
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
			Volumes:       volumes,
		},
	}
	_, err := cl.Pods(api.NamespaceDefault).Create(pod)
	Expect(err).To(BeNil(), fmt.Sprintf("Error creating Pod %v", err))
}
