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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	apiUnversioned "k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"
	"k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Summary API", func() {
	f := NewDefaultFramework("summary-test")
	Context("when querying /stats/summary", func() {
		It("it should report resource usage through the stats api", func() {
			const pod0 = "stats-busybox-0"
			const pod1 = "stats-busybox-1"

			By("Creating test pods")
			createSummaryTestPods(f, pod0, pod1)

			// Setup expectations
			lower := lowerBound
			lower.Pods = []stats.PodStats{
				namedPod(f.Namespace.Name, pod0, podLower),
				namedPod(f.Namespace.Name, pod1, podLower),
			}
			upper := upperBound
			upper.Pods = []stats.PodStats{
				namedPod(f.Namespace.Name, pod0, podUpper),
				namedPod(f.Namespace.Name, pod1, podUpper),
			}

			By("Returning stats summary")
			summary := stats.Summary{}
			Eventually(func() error {
				resp, err := http.Get(*kubeletAddress + "/stats/summary")
				if err != nil {
					return fmt.Errorf("Failed to get /stats/summary - %v", err)
				}
				contentsBytes, err := ioutil.ReadAll(resp.Body)
				if err != nil {
					return fmt.Errorf("Failed to read /stats/summary - %+v", resp)
				}
				contents := string(contentsBytes)
				decoder := json.NewDecoder(strings.NewReader(contents))
				err = decoder.Decode(&summary)
				if err != nil {
					return fmt.Errorf("Failed to parse /stats/summary to go struct: %+v", resp)
				}
				errs := checkSummary(summary, lowerBound, upperBound)
				if len(errs) > 0 {
					return errors.NewAggregate(errs)
				}
				return nil
			}, 1*time.Minute, time.Second*15).Should(BeNil())
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
	f.CreatePods(pods)
}

const (
	kb = 1000
	mb = 1000 * kb
	gb = 1000 * mb
)

var (
	podLower = stats.PodStats{
		Containers: []stats.ContainerStats{
			{
				Name: "busybox-container",
				CPU: &stats.CPUStats{
					UsageNanoCores:       val(100000),
					UsageCoreNanoSeconds: val(10000000),
				},
				Memory: &stats.MemoryStats{
					AvailableBytes:  val(1 * mb),
					UsageBytes:      val(10 * kb),
					WorkingSetBytes: val(10 * kb),
					RSSBytes:        val(1 * kb),
					PageFaults:      val(100),
					MajorPageFaults: val(0),
				},
				Rootfs: &stats.FsStats{
					AvailableBytes: val(100 * mb),
					CapacityBytes:  val(100 * mb),
					UsedBytes:      val(kb),
				},
				Logs: &stats.FsStats{
					AvailableBytes: val(100 * mb),
					CapacityBytes:  val(100 * mb),
					UsedBytes:      val(kb),
				},
			},
		},
		Network: &stats.NetworkStats{
			RxBytes:  val(10),
			RxErrors: val(0),
			TxBytes:  val(10),
			TxErrors: val(0),
		},
		VolumeStats: []stats.VolumeStats{{
			Name: "test-empty-dir",
			FsStats: stats.FsStats{
				AvailableBytes: val(100 * mb),
				CapacityBytes:  val(100 * mb),
				UsedBytes:      val(kb),
			},
		}},
	}

	lowerBound = stats.Summary{
		Node: stats.NodeStats{
			SystemContainers: []stats.ContainerStats{
				{
					Name: "kubelet",
					CPU: &stats.CPUStats{
						UsageNanoCores:       val(100000),
						UsageCoreNanoSeconds: val(10000000),
					},
					Memory: &stats.MemoryStats{
						AvailableBytes:  val(100 * mb),
						UsageBytes:      val(10 * mb),
						WorkingSetBytes: val(10 * mb),
						RSSBytes:        val(10 * mb),
						PageFaults:      val(1000),
						MajorPageFaults: val(0),
					},
					Rootfs: &stats.FsStats{
						AvailableBytes: val(100 * mb),
						CapacityBytes:  val(100 * mb),
						UsedBytes:      val(0),
					},
					Logs: &stats.FsStats{
						AvailableBytes: val(100 * mb),
						CapacityBytes:  val(100 * mb),
						UsedBytes:      val(kb),
					},
				},
				{
					Name: "runtime",
					CPU: &stats.CPUStats{
						UsageNanoCores:       val(100000),
						UsageCoreNanoSeconds: val(10000000),
					},
					Memory: &stats.MemoryStats{
						AvailableBytes:  val(100 * mb),
						UsageBytes:      val(100 * mb),
						WorkingSetBytes: val(10 * mb),
						RSSBytes:        val(10 * mb),
						PageFaults:      val(100000),
						MajorPageFaults: val(0),
					},
					Rootfs: &stats.FsStats{
						AvailableBytes: val(100 * mb),
						CapacityBytes:  val(100 * mb),
						UsedBytes:      val(0),
					},
					Logs: &stats.FsStats{
						AvailableBytes: val(100 * mb),
						CapacityBytes:  val(100 * mb),
						UsedBytes:      val(kb),
					},
				},
			},
			CPU: &stats.CPUStats{
				UsageNanoCores:       val(100000),
				UsageCoreNanoSeconds: val(1000000000),
			},
			Memory: &stats.MemoryStats{
				AvailableBytes:  val(100 * mb),
				UsageBytes:      val(10 * mb),
				WorkingSetBytes: val(10 * mb),
				RSSBytes:        val(1 * mb),
				PageFaults:      val(1000),
				MajorPageFaults: val(0),
			},
			Network: &stats.NetworkStats{
				RxBytes:  val(1 * mb),
				RxErrors: val(0),
				TxBytes:  val(10 * kb),
				TxErrors: val(0),
			},
			Fs: &stats.FsStats{
				AvailableBytes: val(100 * mb),
				CapacityBytes:  val(100 * mb),
				UsedBytes:      val(kb),
				InodesFree:     val(1E4),
			},
			Runtime: &stats.RuntimeStats{
				ImageFs: &stats.FsStats{
					AvailableBytes: val(100 * mb),
					CapacityBytes:  val(100 * mb),
					UsedBytes:      val(kb),
					InodesFree:     val(1E4),
				},
			},
		},
	}

	podUpper = stats.PodStats{
		Containers: []stats.ContainerStats{
			{
				Name: "busybox-container",
				CPU: &stats.CPUStats{
					UsageNanoCores:       val(100000000),
					UsageCoreNanoSeconds: val(1000000000),
				},
				Memory: &stats.MemoryStats{
					AvailableBytes:  val(10 * mb),
					UsageBytes:      val(mb),
					WorkingSetBytes: val(mb),
					RSSBytes:        val(mb),
					PageFaults:      val(100000),
					MajorPageFaults: val(10),
				},
				Rootfs: &stats.FsStats{
					AvailableBytes: val(100 * gb),
					CapacityBytes:  val(100 * gb),
					UsedBytes:      val(10 * mb),
				},
				Logs: &stats.FsStats{
					AvailableBytes: val(100 * gb),
					CapacityBytes:  val(100 * gb),
					UsedBytes:      val(10 * mb),
				},
			},
		},
		Network: &stats.NetworkStats{
			RxBytes:  val(10 * mb),
			RxErrors: val(1000),
			TxBytes:  val(10 * mb),
			TxErrors: val(1000),
		},
		VolumeStats: []stats.VolumeStats{{
			Name: "test-empty-dir",
			FsStats: stats.FsStats{
				AvailableBytes: val(100 * gb),
				CapacityBytes:  val(100 * gb),
				UsedBytes:      val(1 * mb),
			},
		}},
	}

	upperBound = stats.Summary{
		Node: stats.NodeStats{
			SystemContainers: []stats.ContainerStats{
				{
					Name: "kubelet",
					CPU: &stats.CPUStats{
						UsageNanoCores:       val(2E9),
						UsageCoreNanoSeconds: val(10E12),
					},
					Memory: &stats.MemoryStats{
						AvailableBytes:  val(100 * gb),
						UsageBytes:      val(10 * gb),
						WorkingSetBytes: val(1 * gb),
						RSSBytes:        val(1 * gb),
						PageFaults:      val(1E9),
						MajorPageFaults: val(100000),
					},
					Rootfs: &stats.FsStats{
						AvailableBytes: val(100 * gb),
						CapacityBytes:  val(100 * gb),
						UsedBytes:      val(0), // Kubelet doesn't write.
					},
					Logs: &stats.FsStats{
						AvailableBytes: val(100 * gb),
						CapacityBytes:  val(100 * gb),
						UsedBytes:      val(10 * gb),
					},
				},
				{
					Name: "runtime",
					CPU: &stats.CPUStats{
						UsageNanoCores:       val(2E9),
						UsageCoreNanoSeconds: val(10E12),
					},
					Memory: &stats.MemoryStats{
						AvailableBytes:  val(100 * gb),
						UsageBytes:      val(10 * gb),
						WorkingSetBytes: val(1 * gb),
						RSSBytes:        val(1 * gb),
						PageFaults:      val(1E9),
						MajorPageFaults: val(100000),
					},
					Rootfs: &stats.FsStats{
						AvailableBytes: val(100 * gb),
						CapacityBytes:  val(100 * gb),
						UsedBytes:      val(10 * gb),
					},
					Logs: &stats.FsStats{
						AvailableBytes: val(100 * gb),
						CapacityBytes:  val(100 * gb),
						UsedBytes:      val(10 * gb),
					},
				},
			},
			CPU: &stats.CPUStats{
				UsageNanoCores:       val(2E9),
				UsageCoreNanoSeconds: val(10E12),
			},
			Memory: &stats.MemoryStats{
				AvailableBytes:  val(100 * gb),
				UsageBytes:      val(10 * gb),
				WorkingSetBytes: val(1 * gb),
				RSSBytes:        val(1 * gb),
				PageFaults:      val(1E9),
				MajorPageFaults: val(100000),
			},
			Network: &stats.NetworkStats{
				RxBytes:  val(100 * gb),
				RxErrors: val(100000),
				TxBytes:  val(10 * gb),
				TxErrors: val(100000),
			},
			Fs: &stats.FsStats{
				AvailableBytes: val(100 * gb),
				CapacityBytes:  val(100 * gb),
				UsedBytes:      val(10 * gb),
				InodesFree:     val(1E6),
			},
			Runtime: &stats.RuntimeStats{
				ImageFs: &stats.FsStats{
					AvailableBytes: val(100 * gb),
					CapacityBytes:  val(100 * gb),
					UsedBytes:      val(10 * gb),
					InodesFree:     val(1E6),
				},
			},
		},
	}

	ignoredFields = sets.NewString(
		"Name",
		"NodeName",
		"PodRef",
		"StartTime",
		"UserDefinedMetrics",
	)

	allowedNils = sets.NewString(
		".Node.SystemContainers[kubelet].Memory.AvailableBytes",
		".Node.SystemContainers[runtime].Memory.AvailableBytes",
		// TODO(#28395): Figure out why UsedBytes is nil on ubuntu-trusty-docker10 and coreos-stable20160622
		".Node.SystemContainers[kubelet].Rootfs.UsedBytes",
		".Node.SystemContainers[kubelet].Logs.UsedBytes",
		".Node.SystemContainers[runtime].Rootfs.UsedBytes",
		".Node.SystemContainers[runtime].Logs.UsedBytes",
		// TODO: Handle non-eth0 network interface names.
		".Node.Network",
	)
)

func checkSummary(actual, lower, upper stats.Summary) []error {
	return checkValue("", reflect.ValueOf(actual), reflect.ValueOf(lower), reflect.ValueOf(upper))
}

func checkValue(name string, actual, lower, upper reflect.Value) (errs []error) {
	// Provide more useful error messages in the case of a panic.
	defer func() {
		if err := recover(); err != nil {
			errs = append(errs, fmt.Errorf("panic checking %s (%v): %v", name, actual, err))
		}
	}()

	if !actual.IsValid() {
		if !lower.IsValid() {
			// Expected zero-value, ignore it.
			return nil
		}
		return []error{fmt.Errorf("%s is an unexpected zero-value!", name)}
	}

	switch actual.Kind() {

	case reflect.Struct:
		typ := actual.Type()
		for i := 0; i < actual.NumField(); i++ {
			fieldName := typ.Field(i).Name
			if ignoredFields.Has(fieldName) {
				continue
			}

			name := name + "." + typ.Field(i).Name
			switch fieldName {
			case "Time":
				// Special-case timestamp fields.
				errs = append(errs, checkTime(name, actual.Field(i).Interface().(apiUnversioned.Time))...)
			default:
				errs = append(errs, checkValue(name, actual.Field(i), lower.Field(i), upper.Field(i))...)
			}
		}

	case reflect.Slice:
		if actual.Type().Name() == "VolumeStats" && actual.Len() != lower.Len() {
			errs = append(errs, fmt.Errorf("%s length mismatch! expected: %d, actual: %d", name, lower.Len(), actual.Len()))
		}
		for i := 0; i < lower.Len(); i++ {
			actualIndex, err := findMatch(name, lower.Index(i), actual)
			if err != nil {
				errs = append(errs, err)
				continue
			}

			name := fmt.Sprintf("%s[%s]", name, summaryObjectID(lower.Index(i)))
			result := checkValue(name, actual.Index(actualIndex), lower.Index(i), upper.Index(i))
			errs = append(errs, result...)
		}

	case reflect.Uint64:
		if actual.Uint() < lower.Uint() {
			errs = append(errs, fmt.Errorf("%s is too small! %d < %d", name, actual.Uint(), lower.Uint()))
		} else if actual.Uint() > upper.Uint() {
			errs = append(errs, fmt.Errorf("%s is too big! %d > %d", name, actual.Uint(), upper.Uint()))
		}

	case reflect.Ptr:
		if actual.IsNil() {
			if !allowedNils.Has(name) {
				errs = append(errs, fmt.Errorf("%s is nil!", name))
			}
		} else {
			errs = append(errs, checkValue(name, actual.Elem(), lower.Elem(), upper.Elem())...)
		}

	default:
		errs = append(errs, fmt.Errorf("%s is an unhandled type (%s)", name, actual.Type().Name()))
	}

	return errs
}

func checkTime(name string, t apiUnversioned.Time) []error {
	const maxInterval = time.Minute
	now := time.Now()
	if now.Sub(t.Time) > maxInterval {
		return []error{fmt.Errorf("%s is too old! now: %v, then: %v", name, now, t)}
	} else if t.After(now) {
		return []error{fmt.Errorf("%s is in the future! now: %v, then: %v", name, now, t)}
	}
	return nil
}

func findMatch(name string, targetValue, sliceValue reflect.Value) (int, error) {
	targetID := summaryObjectID(targetValue)
	for i := 0; i < sliceValue.Len(); i++ {
		id := summaryObjectID(sliceValue.Index(i))
		if id == targetID {
			return i, nil
		}
	}
	return -1, fmt.Errorf("%s missing object %s", name, targetID)
}

func summaryObjectID(value reflect.Value) string {
	switch v := value.Interface().(type) {
	case stats.PodStats:
		return fmt.Sprintf("%s::%s", v.PodRef.Namespace, v.PodRef.Name)
	case stats.ContainerStats:
		return v.Name
	case stats.VolumeStats:
		return v.Name
	case stats.UserDefinedMetric:
		return v.Name
	default:
		framework.Failf("Unknown type: %+v", v)
		return "???"
	}
}

// Helpers for setting up bounds
func val(v uint64) *uint64 {
	return &v
}

func namedPod(namespace, name string, pod stats.PodStats) stats.PodStats {
	pod.PodRef.Name = name
	pod.PodRef.Namespace = namespace
	return pod
}
