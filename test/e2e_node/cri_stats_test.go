/*
Copyright The Kubernetes Authors.

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
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/onsi/gomega/types"

	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("CRIContainerMetrics", "[LinuxOnly]", framework.WithNodeConformance(), framework.WithFeatureGate(features.PodAndContainerStatsFromCRI), func() {
	f := framework.NewDefaultFramework("cri-container-metrics")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	ginkgo.Context("when querying /metrics/cadvisor", func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			if !e2eskipper.IsFeatureGateEnabled(features.PodAndContainerStatsFromCRI) {
				e2eskipper.Skipf("requires PodAndContainerStatsFromCRI feature gate to be enabled")
			}
			createMetricsPods(ctx, f)
		})
		ginkgo.AfterEach(func(ctx context.Context) {
			removeMetricsPods(ctx, f)
		})
		ginkgo.It("should report container metrics from CRI", func(ctx context.Context) {
			// Required metrics: these must exist as metric keys AND contain entries for our pods.
			requiredKeys := gstruct.Keys{}
			ctrMatches := map[string]types.GomegaMatcher{
				"container_cpu_cfs_periods_total":           boundedSample(0, 100000),
				"container_cpu_cfs_throttled_periods_total": boundedSample(0, 100000),
				"container_cpu_cfs_throttled_seconds_total": boundedSample(0, 100),
				"container_cpu_system_seconds_total":        boundedSample(0, 100),
				"container_cpu_usage_seconds_total":         boundedSample(0, 100),
				"container_cpu_user_seconds_total":          boundedSample(0, 100),
				"container_file_descriptors":                boundedSample(0, 100),
				"container_fs_reads_bytes_total":            boundedSample(0, 100*e2evolume.Mb),
				"container_fs_reads_total":                  boundedSample(0, 10000),
				"container_fs_usage_bytes":                  boundedSample(0, 1*e2evolume.Gb),
				"container_fs_writes_bytes_total":           boundedSample(0, 100*e2evolume.Mb),
				"container_fs_writes_total":                 boundedSample(0, 100000),
				"container_last_seen":                       boundedSample(time.Now().Add(-maxStatsAge).Unix(), time.Now().Add(2*time.Minute).Unix()),
				"container_memory_cache":                    boundedSample(0, 80*e2evolume.Mb),
				"container_memory_failures_total":           boundedSample(0, 1000000),
				"container_memory_mapped_file":              boundedSample(0, 80*e2evolume.Mb),
				"container_memory_max_usage_bytes":          boundedSample(0, 80*e2evolume.Mb),
				"container_memory_rss":                      boundedSample(10*e2evolume.Kb, 80*e2evolume.Mb),
				"container_memory_swap":                     preciseSample(0),
				"container_memory_usage_bytes":              boundedSample(10*e2evolume.Kb, 80*e2evolume.Mb),
				"container_memory_working_set_bytes":        boundedSample(10*e2evolume.Kb, 80*e2evolume.Mb),
				"container_oom_events_total":                preciseSample(0),
				"container_processes":                       boundedSample(0, 10),
				"container_sockets":                         boundedSample(0, 10),
				"container_spec_cpu_period":                 preciseSample(100000),
				"container_spec_cpu_shares":                 preciseSample(2),
				// CRI runtimes may report the exact memory limit (80M = 80000000) without
				// page-alignment; cadvisor rounds down to page size. Accept both.
				"container_spec_memory_limit_bytes":      boundedSample(79953920, 80*e2evolume.Mb),
				"container_spec_memory_swap_limit_bytes": boundedSample(0, 80*e2evolume.Mb),
				"container_start_time_seconds":           boundedSample(time.Now().Add(-maxStatsAge).Unix(), time.Now().Add(2*time.Minute).Unix()),
				"container_threads":                      boundedSample(0, 10),
			}
			appendMatchesForContainer(f.Namespace.Name, pod0, pod1, "busybox-container", ctrMatches, requiredKeys, gstruct.AllowDuplicates|gstruct.IgnoreExtras)

			// Optional metrics: the metric key itself may be absent (not all runtimes
			// implement every metric family). Use a separate keys map validated with
			// IgnoreMissing at the MatchKeys level.
			optionalKeys := gstruct.Keys{}
			ctrOptionalMatches := map[string]types.GomegaMatcher{
				// blkio metrics availability depends on kernel/runtime support
				"container_blkio_device_usage_total": boundedSample(0, 10000000),
				// The following fs metrics may not be available depending on the filesystem driver
				"container_fs_io_current":                     boundedSample(0, 100),
				"container_fs_io_time_seconds_total":          boundedSample(0, 100),
				"container_fs_io_time_weighted_seconds_total": boundedSample(0, 100),
				// CRI reports filesystem-level inode counts (whole partition), not per-container.
				"container_fs_inodes_free":         boundedSample(0, 10*e2evolume.Mb),
				"container_fs_inodes_total":        boundedSample(0, 10*e2evolume.Mb),
				"container_fs_limit_bytes":         boundedSample(100*e2evolume.Mb, 10*e2evolume.Tb),
				"container_fs_read_seconds_total":  preciseSample(0),
				"container_fs_reads_merged_total":  preciseSample(0),
				"container_fs_sector_reads_total":  preciseSample(0),
				"container_fs_sector_writes_total": preciseSample(0),
				"container_fs_write_seconds_total": preciseSample(0),
				"container_fs_writes_merged_total": preciseSample(0),
				// failcnt and kernel_usage are only available on cgroup v1
				"container_memory_failcnt":      preciseSample(0),
				"container_memory_kernel_usage": boundedSample(0, 80*e2evolume.Mb),
				// hugetlb metrics not implemented by all runtimes
				"container_hugetlb_max_usage_bytes": boundedSample(0, 10*e2evolume.Mb),
				"container_hugetlb_usage_bytes":     boundedSample(0, 10*e2evolume.Mb),
				// cpu quota may not be set
				"container_spec_cpu_quota": boundedSample(0, 100000),
				// not all runtimes report these
				"container_spec_memory_reservation_limit_bytes": preciseSample(0),
				"container_ulimits_soft":                        boundedSample(0, 1073741816),
				// PSI metrics require kernel support and runtime implementation
				"container_pressure_cpu_stalled_seconds_total":    boundedSample(0, 100),
				"container_pressure_cpu_waiting_seconds_total":    boundedSample(0, 100),
				"container_pressure_io_stalled_seconds_total":     boundedSample(0, 100),
				"container_pressure_io_waiting_seconds_total":     boundedSample(0, 100),
				"container_pressure_memory_stalled_seconds_total": boundedSample(0, 100),
				"container_pressure_memory_waiting_seconds_total": boundedSample(0, 100),
			}
			appendMatchesForContainer(f.Namespace.Name, pod0, pod1, "busybox-container", ctrOptionalMatches, optionalKeys, gstruct.AllowDuplicates|gstruct.IgnoreMissing|gstruct.IgnoreExtras)

			podMatches := map[string]types.GomegaMatcher{
				// Network counters are cumulative since the pod sandbox was created and can
				// be large on long-running nodes; use generous upper bounds.
				"container_network_receive_bytes_total":            boundedSample(0, 10*e2evolume.Gb),
				"container_network_receive_errors_total":           boundedSample(0, 1000000),
				"container_network_receive_packets_dropped_total":  boundedSample(0, 1000000),
				"container_network_receive_packets_total":          boundedSample(0, 1000000000),
				"container_network_transmit_bytes_total":           boundedSample(0, 10*e2evolume.Gb),
				"container_network_transmit_errors_total":          boundedSample(0, 1000000),
				"container_network_transmit_packets_dropped_total": boundedSample(0, 1000000),
				"container_network_transmit_packets_total":         boundedSample(0, 1000000000),
			}
			// Network metrics are reported on the pod sandbox container.
			// IgnoreMissing at both levels: containerd may not expose network metrics via CRI.
			appendMatchesForContainer(f.Namespace.Name, pod0, pod1, "POD", podMatches, optionalKeys, gstruct.AllowDuplicates|gstruct.IgnoreMissing|gstruct.IgnoreExtras)

			matchRequired := gstruct.MatchKeys(gstruct.IgnoreExtras, requiredKeys)
			matchOptional := gstruct.MatchKeys(gstruct.IgnoreExtras|gstruct.IgnoreMissing, optionalKeys)
			matchResourceMetrics := gomega.And(matchRequired, matchOptional)

			ginkgo.By("Giving pods time to start up and produce metrics")
			gomega.Eventually(ctx, getContainerMetrics, 2*time.Minute, 15*time.Second).Should(matchResourceMetrics)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(ctx, getContainerMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
		})

		ginkgo.It("should report kubelet_metrics_provider with provider cri", func(ctx context.Context) {
			ginkgo.By("Checking kubelet exposes kubelet_metrics_provider metric with provider=cri")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				metrics, err := e2emetrics.GrabKubeletMetricsWithoutProxy(ctx, fmt.Sprintf("%s:%d", nodeNameOrIP(), ports.KubeletReadOnlyPort), "/metrics")
				if err != nil {
					return err
				}
				samples, ok := metrics["kubelet_metrics_provider"]
				if !ok {
					return fmt.Errorf("kubelet_metrics_provider metric not found")
				}
				for _, sample := range samples {
					if string(sample.Metric["provider"]) == "cri" {
						return nil
					}
				}
				return fmt.Errorf("kubelet_metrics_provider with provider=cri not found, got: %v", samples)
			}, 1*time.Minute, 15*time.Second).Should(gomega.Succeed())
		})
	})
})
