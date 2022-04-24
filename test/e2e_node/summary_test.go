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
	"fmt"
	"os"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletstatsv1alpha1 "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	admissionapi "k8s.io/pod-security-admission/api"

	systemdutil "github.com/coreos/go-systemd/v22/util"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/onsi/gomega/types"
)

var _ = SIGDescribe("Summary API [NodeConformance]", func() {
	f := framework.NewDefaultFramework("summary-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	ginkgo.Context("when querying /stats/summary", func() {
		ginkgo.AfterEach(func() {
			if !ginkgo.CurrentSpecReport().Failed() {
				return
			}
			if framework.TestContext.DumpLogsOnFailure {
				e2ekubectl.LogFailedContainers(f.ClientSet, f.Namespace.Name, framework.Logf)
			}
			ginkgo.By("Recording processes in system cgroups")
			recordSystemCgroupProcesses()
		})
		ginkgo.It("should report resource usage through the stats api", func() {
			const pod0 = "stats-busybox-0"
			const pod1 = "stats-busybox-1"

			ginkgo.By("Creating test pods")
			numRestarts := int32(1)
			pods := getSummaryTestPods(f, numRestarts, pod0, pod1)
			f.PodClient().CreateBatch(pods)

			ginkgo.By("restarting the containers to ensure container metrics are still being gathered after a container is restarted")
			gomega.Eventually(func() error {
				for _, pod := range pods {
					err := verifyPodRestartCount(f, pod.Name, len(pod.Spec.Containers), numRestarts)
					if err != nil {
						return err
					}
				}
				return nil
			}, time.Minute, 5*time.Second).Should(gomega.BeNil())

			ginkgo.By("Waiting 15 seconds for cAdvisor to collect 2 stats points")
			time.Sleep(15 * time.Second)

			// Setup expectations.
			const (
				maxStartAge = time.Hour * 24 * 365 // 1 year
				maxStatsAge = time.Minute
			)
			ginkgo.By("Fetching node so we can match against an appropriate memory limit")
			node := getLocalNode(f)
			memoryCapacity := node.Status.Capacity["memory"]
			memoryLimit := memoryCapacity.Value()
			fsCapacityBounds := bounded(100*e2evolume.Mb, 10*e2evolume.Tb)
			// Expectations for system containers.
			sysContExpectations := func() types.GomegaMatcher {
				return gstruct.MatchAllFields(gstruct.Fields{
					"Name":      gstruct.Ignore(),
					"StartTime": recent(maxStartAge),
					"CPU": ptrMatchAllFields(gstruct.Fields{
						"Time": recent(maxStatsAge),
						// CRI stats provider tries to estimate the value of UsageNanoCores. This value can be
						// either 0 or between 10000 and 2e9.
						// Please refer, https://github.com/kubernetes/kubernetes/pull/95345#discussion_r501630942
						// for more information.
						"UsageNanoCores":       gomega.SatisfyAny(gstruct.PointTo(gomega.BeZero()), bounded(10000, 2e9)),
						"UsageCoreNanoSeconds": bounded(10000000, 1e15),
					}),
					"Memory": ptrMatchAllFields(gstruct.Fields{
						"Time": recent(maxStatsAge),
						// We don't limit system container memory.
						"AvailableBytes":  gomega.BeNil(),
						"UsageBytes":      bounded(1*e2evolume.Mb, memoryLimit),
						"WorkingSetBytes": bounded(1*e2evolume.Mb, memoryLimit),
						// this now returns /sys/fs/cgroup/memory.stat total_rss
						"RSSBytes":        bounded(1*e2evolume.Mb, memoryLimit),
						"PageFaults":      bounded(1000, 1e9),
						"MajorPageFaults": bounded(0, 100000),
					}),
					"Accelerators":       gomega.BeEmpty(),
					"Rootfs":             gomega.BeNil(),
					"Logs":               gomega.BeNil(),
					"UserDefinedMetrics": gomega.BeEmpty(),
				})
			}
			expectedPageFaultsUpperBound := 1000000
			expectedMajorPageFaultsUpperBound := 15
			if IsCgroup2UnifiedMode() {
				// On cgroupv2 these stats are recursive, so make sure they are at least like the value set
				// above for the container.
				expectedPageFaultsUpperBound = 1e9
				expectedMajorPageFaultsUpperBound = 100000
			}

			podsContExpectations := sysContExpectations().(*gstruct.FieldsMatcher)
			podsContExpectations.Fields["Memory"] = ptrMatchAllFields(gstruct.Fields{
				"Time": recent(maxStatsAge),
				// Pods are limited by Node Allocatable
				"AvailableBytes":  bounded(1*e2evolume.Kb, memoryLimit),
				"UsageBytes":      bounded(10*e2evolume.Kb, memoryLimit),
				"WorkingSetBytes": bounded(10*e2evolume.Kb, memoryLimit),
				"RSSBytes":        bounded(1*e2evolume.Kb, memoryLimit),
				"PageFaults":      bounded(0, expectedPageFaultsUpperBound),
				"MajorPageFaults": bounded(0, expectedMajorPageFaultsUpperBound),
			})
			runtimeContExpectations := sysContExpectations().(*gstruct.FieldsMatcher)
			systemContainers := gstruct.Elements{
				"kubelet": sysContExpectations(),
				"runtime": runtimeContExpectations,
				"pods":    podsContExpectations,
			}
			// The Kubelet only manages the 'misc' system container if the host is not running systemd.
			if !systemdutil.IsRunningSystemd() {
				framework.Logf("Host not running systemd; expecting 'misc' system container.")
				miscContExpectations := sysContExpectations().(*gstruct.FieldsMatcher)
				// Misc processes are system-dependent, so relax the memory constraints.
				miscContExpectations.Fields["Memory"] = ptrMatchAllFields(gstruct.Fields{
					"Time": recent(maxStatsAge),
					// We don't limit system container memory.
					"AvailableBytes":  gomega.BeNil(),
					"UsageBytes":      bounded(100*e2evolume.Kb, memoryLimit),
					"WorkingSetBytes": bounded(100*e2evolume.Kb, memoryLimit),
					"RSSBytes":        bounded(100*e2evolume.Kb, memoryLimit),
					"PageFaults":      bounded(1000, 1e9),
					"MajorPageFaults": bounded(0, 100000),
				})
				systemContainers["misc"] = miscContExpectations
			}
			// Expectations for pods.
			podExpectations := gstruct.MatchAllFields(gstruct.Fields{
				"PodRef":    gstruct.Ignore(),
				"StartTime": recent(maxStartAge),
				"Containers": gstruct.MatchAllElements(summaryObjectID, gstruct.Elements{
					"busybox-container": gstruct.MatchAllFields(gstruct.Fields{
						"Name":      gomega.Equal("busybox-container"),
						"StartTime": recent(maxStartAge),
						"CPU": ptrMatchAllFields(gstruct.Fields{
							"Time":                 recent(maxStatsAge),
							"UsageNanoCores":       bounded(10000, 1e9),
							"UsageCoreNanoSeconds": bounded(10000000, 1e11),
						}),
						"Memory": ptrMatchAllFields(gstruct.Fields{
							"Time":            recent(maxStatsAge),
							"AvailableBytes":  bounded(1*e2evolume.Kb, 80*e2evolume.Mb),
							"UsageBytes":      bounded(10*e2evolume.Kb, 80*e2evolume.Mb),
							"WorkingSetBytes": bounded(10*e2evolume.Kb, 80*e2evolume.Mb),
							"RSSBytes":        bounded(1*e2evolume.Kb, 80*e2evolume.Mb),
							"PageFaults":      bounded(100, expectedPageFaultsUpperBound),
							"MajorPageFaults": bounded(0, expectedMajorPageFaultsUpperBound),
						}),
						"Accelerators": gomega.BeEmpty(),
						"Rootfs": ptrMatchAllFields(gstruct.Fields{
							"Time":           recent(maxStatsAge),
							"AvailableBytes": fsCapacityBounds,
							"CapacityBytes":  fsCapacityBounds,
							"UsedBytes":      bounded(e2evolume.Kb, 10*e2evolume.Mb),
							"InodesFree":     bounded(1e4, 1e8),
							"Inodes":         bounded(1e4, 1e8),
							"InodesUsed":     bounded(0, 1e8),
						}),
						"Logs": ptrMatchAllFields(gstruct.Fields{
							"Time":           recent(maxStatsAge),
							"AvailableBytes": fsCapacityBounds,
							"CapacityBytes":  fsCapacityBounds,
							"UsedBytes":      bounded(e2evolume.Kb, 10*e2evolume.Mb),
							"InodesFree":     bounded(1e4, 1e8),
							"Inodes":         bounded(1e4, 1e8),
							"InodesUsed":     bounded(0, 1e8),
						}),
						"UserDefinedMetrics": gomega.BeEmpty(),
					}),
				}),
				"Network": ptrMatchAllFields(gstruct.Fields{
					"Time": recent(maxStatsAge),
					"InterfaceStats": gstruct.MatchAllFields(gstruct.Fields{
						"Name":     gomega.Equal("eth0"),
						"RxBytes":  bounded(10, 10*e2evolume.Mb),
						"RxErrors": bounded(0, 1000),
						"TxBytes":  bounded(10, 10*e2evolume.Mb),
						"TxErrors": bounded(0, 1000),
					}),
					"Interfaces": gomega.Not(gomega.BeNil()),
				}),
				"CPU": ptrMatchAllFields(gstruct.Fields{
					"Time":                 recent(maxStatsAge),
					"UsageNanoCores":       bounded(10000, 1e9),
					"UsageCoreNanoSeconds": bounded(10000000, 1e11),
				}),
				"Memory": ptrMatchAllFields(gstruct.Fields{
					"Time":            recent(maxStatsAge),
					"AvailableBytes":  bounded(1*e2evolume.Kb, 80*e2evolume.Mb),
					"UsageBytes":      bounded(10*e2evolume.Kb, 80*e2evolume.Mb),
					"WorkingSetBytes": bounded(10*e2evolume.Kb, 80*e2evolume.Mb),
					"RSSBytes":        bounded(1*e2evolume.Kb, 80*e2evolume.Mb),
					"PageFaults":      bounded(0, expectedPageFaultsUpperBound),
					"MajorPageFaults": bounded(0, expectedMajorPageFaultsUpperBound),
				}),
				"VolumeStats": gstruct.MatchAllElements(summaryObjectID, gstruct.Elements{
					"test-empty-dir": gstruct.MatchAllFields(gstruct.Fields{
						"Name":              gomega.Equal("test-empty-dir"),
						"PVCRef":            gomega.BeNil(),
						"VolumeHealthStats": gomega.BeNil(),
						"FsStats": gstruct.MatchAllFields(gstruct.Fields{
							"Time":           recent(maxStatsAge),
							"AvailableBytes": fsCapacityBounds,
							"CapacityBytes":  fsCapacityBounds,
							"UsedBytes":      bounded(e2evolume.Kb, 1*e2evolume.Mb),
							"InodesFree":     bounded(1e4, 1e8),
							"Inodes":         bounded(1e4, 1e8),
							"InodesUsed":     bounded(0, 1e8),
						}),
					}),
				}),
				"EphemeralStorage": ptrMatchAllFields(gstruct.Fields{
					"Time":           recent(maxStatsAge),
					"AvailableBytes": fsCapacityBounds,
					"CapacityBytes":  fsCapacityBounds,
					"UsedBytes":      bounded(e2evolume.Kb, 21*e2evolume.Mb),
					"InodesFree":     bounded(1e4, 1e8),
					"Inodes":         bounded(1e4, 1e8),
					"InodesUsed":     bounded(0, 1e8),
				}),
				"ProcessStats": ptrMatchAllFields(gstruct.Fields{
					"ProcessCount": bounded(0, 1e8),
				}),
			})

			matchExpectations := ptrMatchAllFields(gstruct.Fields{
				"Node": gstruct.MatchAllFields(gstruct.Fields{
					"NodeName":         gomega.Equal(framework.TestContext.NodeName),
					"StartTime":        recent(maxStartAge),
					"SystemContainers": gstruct.MatchAllElements(summaryObjectID, systemContainers),
					"CPU": ptrMatchAllFields(gstruct.Fields{
						"Time":                 recent(maxStatsAge),
						"UsageNanoCores":       bounded(100e3, 2e9),
						"UsageCoreNanoSeconds": bounded(1e9, 1e15),
					}),
					"Memory": ptrMatchAllFields(gstruct.Fields{
						"Time":            recent(maxStatsAge),
						"AvailableBytes":  bounded(100*e2evolume.Mb, memoryLimit),
						"UsageBytes":      bounded(10*e2evolume.Mb, memoryLimit),
						"WorkingSetBytes": bounded(10*e2evolume.Mb, memoryLimit),
						// this now returns /sys/fs/cgroup/memory.stat total_rss
						"RSSBytes":        bounded(1*e2evolume.Kb, memoryLimit),
						"PageFaults":      bounded(1000, 1e9),
						"MajorPageFaults": bounded(0, 100000),
					}),
					// TODO(#28407): Handle non-eth0 network interface names.
					"Network": ptrMatchAllFields(gstruct.Fields{
						"Time": recent(maxStatsAge),
						"InterfaceStats": gstruct.MatchAllFields(gstruct.Fields{
							"Name":     gomega.Or(gomega.BeEmpty(), gomega.Equal("eth0")),
							"RxBytes":  gomega.Or(gomega.BeNil(), bounded(1*e2evolume.Mb, 100*e2evolume.Gb)),
							"RxErrors": gomega.Or(gomega.BeNil(), bounded(0, 100000)),
							"TxBytes":  gomega.Or(gomega.BeNil(), bounded(10*e2evolume.Kb, 10*e2evolume.Gb)),
							"TxErrors": gomega.Or(gomega.BeNil(), bounded(0, 100000)),
						}),
						"Interfaces": gomega.Not(gomega.BeNil()),
					}),
					"Fs": ptrMatchAllFields(gstruct.Fields{
						"Time":           recent(maxStatsAge),
						"AvailableBytes": fsCapacityBounds,
						"CapacityBytes":  fsCapacityBounds,
						// we assume we are not running tests on machines more than 10tb of disk
						"UsedBytes":  bounded(e2evolume.Kb, 10*e2evolume.Tb),
						"InodesFree": bounded(1e4, 1e8),
						"Inodes":     bounded(1e4, 1e8),
						"InodesUsed": bounded(0, 1e8),
					}),
					"Runtime": ptrMatchAllFields(gstruct.Fields{
						"ImageFs": ptrMatchAllFields(gstruct.Fields{
							"Time":           recent(maxStatsAge),
							"AvailableBytes": fsCapacityBounds,
							"CapacityBytes":  fsCapacityBounds,
							// we assume we are not running tests on machines more than 10tb of disk
							"UsedBytes":  bounded(e2evolume.Kb, 10*e2evolume.Tb),
							"InodesFree": bounded(1e4, 1e8),
							"Inodes":     bounded(1e4, 1e8),
							"InodesUsed": bounded(0, 1e8),
						}),
					}),
					"Rlimit": ptrMatchAllFields(gstruct.Fields{
						"Time":                  recent(maxStatsAge),
						"MaxPID":                bounded(0, 1e8),
						"NumOfRunningProcesses": bounded(0, 1e8),
					}),
				}),
				// Ignore extra pods since the tests run in parallel.
				"Pods": gstruct.MatchElements(summaryObjectID, gstruct.IgnoreExtras, gstruct.Elements{
					fmt.Sprintf("%s::%s", f.Namespace.Name, pod0): podExpectations,
					fmt.Sprintf("%s::%s", f.Namespace.Name, pod1): podExpectations,
				}),
			})

			ginkgo.By("Validating /stats/summary")
			// Give pods a minute to actually start up.
			gomega.Eventually(getNodeSummary, 180*time.Second, 15*time.Second).Should(matchExpectations)
			// Then the summary should match the expectations a few more times.
			gomega.Consistently(getNodeSummary, 30*time.Second, 15*time.Second).Should(matchExpectations)
		})
	})
})

func getSummaryTestPods(f *framework.Framework, numRestarts int32, names ...string) []*v1.Pod {
	pods := make([]*v1.Pod, 0, len(names))
	for _, name := range names {
		pods = append(pods, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyAlways,
				Containers: []v1.Container{
					{
						Name:  "busybox-container",
						Image: busyboxImage,
						SecurityContext: &v1.SecurityContext{
							Capabilities: &v1.Capabilities{
								Add: []v1.Capability{"NET_RAW"},
							},
						},
						Command: getRestartingContainerCommand("/test-empty-dir-mnt", 0, numRestarts, "echo 'some bytes' >/outside_the_volume.txt; ping -c 1 google.com; echo 'hello world' >> /test-empty-dir-mnt/file;"),
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								// Must set memory limit to get MemoryStats.AvailableBytes
								v1.ResourceMemory: resource.MustParse("80M"),
							},
						},
						VolumeMounts: []v1.VolumeMount{
							{MountPath: "/test-empty-dir-mnt", Name: "test-empty-dir"},
						},
					},
				},
				SecurityContext: &v1.PodSecurityContext{
					SELinuxOptions: &v1.SELinuxOptions{
						Level: "s0",
					},
				},
				Volumes: []v1.Volume{
					// TODO(#28393): Test secret volumes
					// TODO(#28394): Test hostpath volumes
					{Name: "test-empty-dir", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}},
				},
			},
		})
	}
	return pods
}

// Mapping function for gstruct.MatchAllElements
func summaryObjectID(element interface{}) string {
	switch el := element.(type) {
	case kubeletstatsv1alpha1.PodStats:
		return fmt.Sprintf("%s::%s", el.PodRef.Namespace, el.PodRef.Name)
	case kubeletstatsv1alpha1.ContainerStats:
		return el.Name
	case kubeletstatsv1alpha1.VolumeStats:
		return el.Name
	case kubeletstatsv1alpha1.UserDefinedMetric:
		return el.Name
	default:
		framework.Failf("Unknown type: %T", el)
		return "???"
	}
}

// Convenience functions for common matcher combinations.
func ptrMatchAllFields(fields gstruct.Fields) types.GomegaMatcher {
	return gstruct.PointTo(gstruct.MatchAllFields(fields))
}

func bounded(lower, upper interface{}) types.GomegaMatcher {
	return gstruct.PointTo(gomega.And(
		gomega.BeNumerically(">=", lower),
		gomega.BeNumerically("<=", upper)))
}

func recent(d time.Duration) types.GomegaMatcher {
	return gomega.WithTransform(func(t metav1.Time) time.Time {
		return t.Time
	}, gomega.And(
		gomega.BeTemporally(">=", time.Now().Add(-d)),
		// Now() is the test start time, not the match time, so permit a few extra minutes.
		gomega.BeTemporally("<", time.Now().Add(3*time.Minute))))
}

func recordSystemCgroupProcesses() {
	cfg, err := getCurrentKubeletConfig()
	if err != nil {
		framework.Logf("Failed to read kubelet config: %v", err)
		return
	}
	cgroups := map[string]string{
		"kubelet": cfg.KubeletCgroups,
		"misc":    cfg.SystemCgroups,
	}
	for name, cgroup := range cgroups {
		if cgroup == "" {
			framework.Logf("Skipping unconfigured cgroup %s", name)
			continue
		}

		filePattern := "/sys/fs/cgroup/cpu/%s/cgroup.procs"
		if IsCgroup2UnifiedMode() {
			filePattern = "/sys/fs/cgroup/%s/cgroup.procs"
		}
		pids, err := os.ReadFile(fmt.Sprintf(filePattern, cgroup))
		if err != nil {
			framework.Logf("Failed to read processes in cgroup %s: %v", name, err)
			continue
		}

		framework.Logf("Processes in %s cgroup (%s):", name, cgroup)
		for _, pid := range strings.Fields(string(pids)) {
			path := fmt.Sprintf("/proc/%s/cmdline", pid)
			cmd, err := os.ReadFile(path)
			if err != nil {
				framework.Logf("  ginkgo.Failed to read %s: %v", path, err)
			} else {
				framework.Logf("  %s", cmd)
			}
		}
	}
}
