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
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"
	"k8s.io/kubernetes/test/e2e/framework"
	m "k8s.io/kubernetes/test/matchers"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/types"
)

var _ = framework.KubeDescribe("Summary API", func() {
	f := framework.NewDefaultFramework("summary-test")
	Context("when querying /stats/summary", func() {
		It("it should report resource usage through the stats api", func() {
			const pod0 = "stats-busybox-0"
			const pod1 = "stats-busybox-1"

			By("Creating test pods")
			createSummaryTestPods(f, pod0, pod1)
			// Wait for cAdvisor to collect 2 stats points
			time.Sleep(15 * time.Second)

			// Setup expectations.
			const (
				kb = 1000
				mb = 1000 * kb
				gb = 1000 * mb
				tb = 1000 * gb

				maxStartAge = time.Hour * 24 * 365 // 1 year
				maxStatsAge = time.Minute
			)
			fsCapacityBounds := bounded(100*mb, 100*gb)
			// Expectations for system containers.
			sysContExpectations := m.StrictStruct(m.Fields{
				"Name":      m.Ignore(),
				"StartTime": m.Recent(maxStartAge),
				"CPU": structP(m.Fields{
					"Time":                 m.Recent(maxStatsAge),
					"UsageNanoCores":       bounded(100000, 2E9),
					"UsageCoreNanoSeconds": bounded(10000000, 1E15),
				}),
				"Memory": structP(m.Fields{
					"Time":            m.Recent(maxStatsAge),
					"AvailableBytes":  bounded(100*mb, 100*gb),
					"UsageBytes":      bounded(10*mb, 1*gb),
					"WorkingSetBytes": bounded(10*mb, 1*gb),
					"RSSBytes":        bounded(10*mb, 1*gb),
					"PageFaults":      bounded(100000, 1E9),
					"MajorPageFaults": bounded(0, 100000),
				}),
				"Rootfs": structP(m.Fields{
					"AvailableBytes": fsCapacityBounds,
					"CapacityBytes":  fsCapacityBounds,
					"UsedBytes":      bounded(0, 10*gb),
					"InodesFree":     bounded(1E4, 1E8),
				}),
				"Logs": structP(m.Fields{
					"AvailableBytes": fsCapacityBounds,
					"CapacityBytes":  fsCapacityBounds,
					"UsedBytes":      bounded(kb, 10*gb),
					"InodesFree":     bounded(1E4, 1E8),
				}),
				"UserDefinedMetrics": BeEmpty(),
			})
			// Expectations for pods.
			podExpectations := m.StrictStruct(m.Fields{
				"PodRef":    m.Ignore(),
				"StartTime": m.Recent(maxStartAge),
				"Containers": m.StrictSlice(summaryObjectID, m.Elements{
					"busybox-container": m.StrictStruct(m.Fields{
						"Name":      Equal("busybox-container"),
						"StartTime": m.Recent(maxStartAge),
						"CPU": structP(m.Fields{
							"Time":                 m.Recent(maxStatsAge),
							"UsageNanoCores":       bounded(100000, 100000000),
							"UsageCoreNanoSeconds": bounded(10000000, 1000000000),
						}),
						"Memory": structP(m.Fields{
							"Time":            m.Recent(maxStatsAge),
							"AvailableBytes":  bounded(1*mb, 10*mb),
							"UsageBytes":      bounded(10*kb, mb),
							"WorkingSetBytes": bounded(10*kb, mb),
							"RSSBytes":        bounded(1*kb, mb),
							"PageFaults":      bounded(100, 100000),
							"MajorPageFaults": bounded(0, 10),
						}),
						"Rootfs": structP(m.Fields{
							"AvailableBytes": fsCapacityBounds,
							"CapacityBytes":  fsCapacityBounds,
							"UsedBytes":      bounded(kb, 10*mb),
							"InodesFree":     bounded(1E4, 1E8),
						}),
						"Logs": structP(m.Fields{
							"AvailableBytes": fsCapacityBounds,
							"CapacityBytes":  fsCapacityBounds,
							"UsedBytes":      bounded(kb, 10*mb),
							"InodesFree":     bounded(1E4, 1E8),
						}),
						"UserDefinedMetrics": BeEmpty(),
					}),
				}),
				"Network": structP(m.Fields{
					"Time":     m.Recent(maxStatsAge),
					"RxBytes":  bounded(10, 10*mb),
					"RxErrors": bounded(0, 1000),
					"TxBytes":  bounded(10, 10*mb),
					"TxErrors": bounded(0, 1000),
				}),
				"VolumeStats": m.StrictSlice(summaryObjectID, m.Elements{
					"test-empty-dir": m.StrictStruct(m.Fields{
						"Name": Equal("test-empty-dir"),
						"FsStats": m.StrictStruct(m.Fields{
							"AvailableBytes": fsCapacityBounds,
							"CapacityBytes":  fsCapacityBounds,
							"UsedBytes":      bounded(kb, 1*mb),
							"InodesFree":     BeNil(),
						}),
					}),
				}),
			})
			matchExpectations := structP(m.Fields{
				"Node": m.StrictStruct(m.Fields{
					"NodeName":  m.Ignore(),
					"StartTime": m.Recent(maxStartAge),
					"SystemContainers": m.StrictSlice(summaryObjectID, m.Elements{
						"kubelet": sysContExpectations,
						"runtime": sysContExpectations,
					}),
					"CPU": structP(m.Fields{
						"Time":                 m.Recent(maxStatsAge),
						"UsageNanoCores":       bounded(100E3, 2E9),
						"UsageCoreNanoSeconds": bounded(1E9, 1E15),
					}),
					"Memory": structP(m.Fields{
						"Time":            m.Recent(maxStatsAge),
						"AvailableBytes":  bounded(100*mb, 100*gb),
						"UsageBytes":      bounded(10*mb, 10*gb),
						"WorkingSetBytes": bounded(10*mb, 1*gb),
						"RSSBytes":        bounded(1*mb, 1*gb),
						"PageFaults":      bounded(1000, 1E9),
						"MajorPageFaults": bounded(0, 100000),
					}),
					// TODO(#28407): Handle non-eth0 network interface names.
					"Network": m.NilOr(
						structP(m.Fields{
							"Time":     m.Recent(maxStatsAge),
							"RxBytes":  bounded(1*mb, 100*gb),
							"RxErrors": bounded(0, 100000),
							"TxBytes":  bounded(10*kb, 10*gb),
							"TxErrors": bounded(0, 100000),
						}),
					),
					"Fs": structP(m.Fields{
						"AvailableBytes": fsCapacityBounds,
						"CapacityBytes":  fsCapacityBounds,
						"UsedBytes":      bounded(kb, 10*gb),
						"InodesFree":     bounded(1E4, 1E8),
					}),
					"Runtime": structP(m.Fields{
						"ImageFs": structP(m.Fields{
							"AvailableBytes": fsCapacityBounds,
							"CapacityBytes":  fsCapacityBounds,
							"UsedBytes":      bounded(kb, 10*gb),
							"InodesFree":     bounded(1E4, 1E8),
						}),
					}),
				}),
				"Pods": m.StrictSlice(summaryObjectID, m.Elements{
					fmt.Sprintf("%s::%s", f.Namespace.Name, pod0): podExpectations,
					fmt.Sprintf("%s::%s", f.Namespace.Name, pod1): podExpectations,
				}),
			})

			By("Validating /stats/summary")
			Eventually(func() *stats.Summary {
				summary, err := getNodeSummary()
				if err != nil {
					framework.Logf("Error retrieving /stats/summary: %v", err)
					return nil
				}
				return summary
			}, 1*time.Minute, time.Second*15).Should(matchExpectations)
		})
	})
})

func createSummaryTestPods(f *framework.Framework, names ...string) {
	pods := make([]*api.Pod, 0, len(names))
	for _, name := range names {
		pods = append(pods, &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: name,
			},
			Spec: api.PodSpec{
				// Don't restart the Pod since it is expected to exit
				RestartPolicy: api.RestartPolicyNever,
				Containers: []api.Container{
					{
						Name:    "busybox-container",
						Image:   ImageRegistry[busyBoxImage],
						Command: []string{"sh", "-c", "while true; do echo 'hello world' | tee /test-empty-dir-mnt/file ; sleep 1; done"},
						Resources: api.ResourceRequirements{
							Limits: api.ResourceList{
								// Must set memory limit to get MemoryStats.AvailableBytes
								api.ResourceMemory: resource.MustParse("10M"),
							},
						},
						VolumeMounts: []api.VolumeMount{
							{MountPath: "/test-empty-dir-mnt", Name: "test-empty-dir"},
						},
					},
				},
				SecurityContext: &api.PodSecurityContext{
					SELinuxOptions: &api.SELinuxOptions{
						Level: "s0",
					},
				},
				Volumes: []api.Volume{
					// TODO(#28393): Test secret volumes
					// TODO(#28394): Test hostpath volumes
					{Name: "test-empty-dir", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				},
			},
		})
	}
	f.PodClient().CreateBatch(pods)
}

func summaryObjectID(element interface{}) string {
	switch el := element.(type) {
	case stats.PodStats:
		return fmt.Sprintf("%s::%s", el.PodRef.Namespace, el.PodRef.Name)
	case stats.ContainerStats:
		return el.Name
	case stats.VolumeStats:
		return el.Name
	case stats.UserDefinedMetric:
		return el.Name
	default:
		framework.Failf("Unknown type: %T", el)
		return "???"
	}
}

// Convenience functions for common matcher combinations.
func structP(fields m.Fields) types.GomegaMatcher {
	return m.Ptr(m.StrictStruct(fields))
}

func bounded(lower, upper interface{}) types.GomegaMatcher {
	return m.Ptr(m.InRange(lower, upper))
}
