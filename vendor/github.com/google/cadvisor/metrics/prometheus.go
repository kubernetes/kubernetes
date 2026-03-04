// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package metrics

import (
	"fmt"
	"regexp"
	"strconv"
	"time"

	"github.com/google/cadvisor/container"
	info "github.com/google/cadvisor/info/v1"
	v2 "github.com/google/cadvisor/info/v2"

	"github.com/prometheus/client_golang/prometheus"

	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

// asFloat64 converts a uint64 into a float64.
func asFloat64(v uint64) float64 { return float64(v) }

// asMicrosecondsToSeconds converts nanoseconds into a float64 representing seconds.
func asMicrosecondsToSeconds(v uint64) float64 {
	return float64(v) / 1e6
}

// asNanosecondsToSeconds converts nanoseconds into a float64 representing seconds.
func asNanosecondsToSeconds(v uint64) float64 {
	return float64(v) / 1e9
}

// fsValues is a helper method for assembling per-filesystem stats.
func fsValues(fsStats []info.FsStats, valueFn func(*info.FsStats) float64, timestamp time.Time) metricValues {
	values := make(metricValues, 0, len(fsStats))
	for _, stat := range fsStats {
		values = append(values, metricValue{
			value:     valueFn(&stat),
			labels:    []string{stat.Device},
			timestamp: timestamp,
		})
	}
	return values
}

// ioValues is a helper method for assembling per-disk and per-filesystem stats.
func ioValues(ioStats []info.PerDiskStats, ioType string, ioValueFn func(uint64) float64,
	fsStats []info.FsStats, valueFn func(*info.FsStats) float64, timestamp time.Time) metricValues {

	values := make(metricValues, 0, len(ioStats)+len(fsStats))
	for _, stat := range ioStats {
		values = append(values, metricValue{
			value:     ioValueFn(stat.Stats[ioType]),
			labels:    []string{stat.Device},
			timestamp: timestamp,
		})
	}
	for _, stat := range fsStats {
		values = append(values, metricValue{
			value:     valueFn(&stat),
			labels:    []string{stat.Device},
			timestamp: timestamp,
		})
	}
	return values
}

// containerMetric describes a multi-dimensional metric used for exposing a
// certain type of container statistic.
type containerMetric struct {
	name        string
	help        string
	valueType   prometheus.ValueType
	extraLabels []string
	condition   func(s info.ContainerSpec) bool
	getValues   func(s *info.ContainerStats) metricValues
}

func (cm *containerMetric) desc(baseLabels []string) *prometheus.Desc {
	return prometheus.NewDesc(cm.name, cm.help, append(baseLabels, cm.extraLabels...), nil)
}

// ContainerLabelsFunc defines all base labels and their values attached to
// each metric exported by cAdvisor.
type ContainerLabelsFunc func(*info.ContainerInfo) map[string]string

// PrometheusCollector implements prometheus.Collector.
type PrometheusCollector struct {
	infoProvider        infoProvider
	errors              prometheus.Gauge
	containerMetrics    []containerMetric
	containerLabelsFunc ContainerLabelsFunc
	includedMetrics     container.MetricSet
	opts                v2.RequestOptions
}

// NewPrometheusCollector returns a new PrometheusCollector. The passed
// ContainerLabelsFunc specifies which base labels will be attached to all
// exported metrics. If left to nil, the DefaultContainerLabels function
// will be used instead.
func NewPrometheusCollector(i infoProvider, f ContainerLabelsFunc, includedMetrics container.MetricSet, now clock.Clock, opts v2.RequestOptions) *PrometheusCollector {
	if f == nil {
		f = DefaultContainerLabels
	}
	c := &PrometheusCollector{
		infoProvider:        i,
		containerLabelsFunc: f,
		errors: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: "container",
			Name:      "scrape_error",
			Help:      "1 if there was an error while getting container metrics, 0 otherwise",
		}),
		containerMetrics: []containerMetric{
			{
				name:      "container_last_seen",
				help:      "Last time a container was seen by the exporter",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{
						value:     float64(now.Now().Unix()),
						timestamp: now.Now(),
					}}
				},
			},
			{
				name:      "container_health_state",
				help:      "The result of the container's health check",
				valueType: prometheus.GaugeValue,
				getValues: getContainerHealthState,
			},
		},
		includedMetrics: includedMetrics,
		opts:            opts,
	}
	if includedMetrics.Has(container.CpuUsageMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:      "container_cpu_user_seconds_total",
				help:      "Cumulative user cpu time consumed in seconds.",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.Cpu.Usage.User) / float64(time.Second),
							timestamp: s.Timestamp,
						},
					}
				},
			}, {
				name:      "container_cpu_system_seconds_total",
				help:      "Cumulative system cpu time consumed in seconds.",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.Cpu.Usage.System) / float64(time.Second),
							timestamp: s.Timestamp,
						},
					}
				},
			}, {
				name:        "container_cpu_usage_seconds_total",
				help:        "Cumulative cpu time consumed in seconds.",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"cpu"},
				getValues: func(s *info.ContainerStats) metricValues {
					if len(s.Cpu.Usage.PerCpu) == 0 {
						if s.Cpu.Usage.Total > 0 {
							return metricValues{{
								value:     float64(s.Cpu.Usage.Total) / float64(time.Second),
								labels:    []string{"total"},
								timestamp: s.Timestamp,
							}}
						}
					}
					values := make(metricValues, 0, len(s.Cpu.Usage.PerCpu))
					for i, value := range s.Cpu.Usage.PerCpu {
						if value > 0 {
							values = append(values, metricValue{
								value:     float64(value) / float64(time.Second),
								labels:    []string{fmt.Sprintf("cpu%02d", i)},
								timestamp: s.Timestamp,
							})
						}
					}
					return values
				},
			}, {
				name:      "container_cpu_cfs_periods_total",
				help:      "Number of elapsed enforcement period intervals.",
				valueType: prometheus.CounterValue,
				condition: func(s info.ContainerSpec) bool { return s.Cpu.Quota != 0 },
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.Cpu.CFS.Periods),
							timestamp: s.Timestamp,
						}}
				},
			}, {
				name:      "container_cpu_cfs_throttled_periods_total",
				help:      "Number of throttled period intervals.",
				valueType: prometheus.CounterValue,
				condition: func(s info.ContainerSpec) bool { return s.Cpu.Quota != 0 },
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.Cpu.CFS.ThrottledPeriods),
							timestamp: s.Timestamp,
						}}
				},
			}, {
				name:      "container_cpu_cfs_throttled_seconds_total",
				help:      "Total time duration the container has been throttled.",
				valueType: prometheus.CounterValue,
				condition: func(s info.ContainerSpec) bool { return s.Cpu.Quota != 0 },
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.Cpu.CFS.ThrottledTime) / float64(time.Second),
							timestamp: s.Timestamp,
						}}
				},
			}, {
				name:      "container_cpu_cfs_burst_periods_total",
				help:      "Number of periods when burst occurs.",
				valueType: prometheus.CounterValue,
				condition: func(s info.ContainerSpec) bool { return s.Cpu.Quota != 0 },
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.Cpu.CFS.BurstsPeriods),
							timestamp: s.Timestamp,
						}}
				},
			}, {
				name:      "container_cpu_cfs_burst_seconds_total",
				help:      "Total time duration the container has been bursted.",
				valueType: prometheus.CounterValue,
				condition: func(s info.ContainerSpec) bool { return s.Cpu.Quota != 0 },
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.Cpu.CFS.BurstTime) / float64(time.Second),
							timestamp: s.Timestamp,
						}}
				},
			},
		}...)
	}
	if includedMetrics.Has(container.ProcessSchedulerMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:      "container_cpu_schedstat_run_seconds_total",
				help:      "Time duration the processes of the container have run on the CPU.",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{
						value:     float64(s.Cpu.Schedstat.RunTime) / float64(time.Second),
						timestamp: s.Timestamp,
					}}
				},
			}, {
				name:      "container_cpu_schedstat_runqueue_seconds_total",
				help:      "Time duration processes of the container have been waiting on a runqueue.",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{
						value:     float64(s.Cpu.Schedstat.RunqueueTime) / float64(time.Second),
						timestamp: s.Timestamp,
					}}
				},
			}, {
				name:      "container_cpu_schedstat_run_periods_total",
				help:      "Number of times processes of the cgroup have run on the cpu",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{
						value:     float64(s.Cpu.Schedstat.RunPeriods),
						timestamp: s.Timestamp,
					}}
				},
			},
		}...)
	}
	if includedMetrics.Has(container.CpuLoadMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:      "container_cpu_load_average_10s",
				help:      "Value of container cpu load average over the last 10 seconds.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Cpu.LoadAverage), timestamp: s.Timestamp}}
				},
			}, {
				name:      "container_cpu_load_d_average_10s",
				help:      "Value of container cpu load.d average over the last 10 seconds.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Cpu.LoadDAverage), timestamp: s.Timestamp}}
				},
			}, {
				name:        "container_tasks_state",
				help:        "Number of tasks in given state",
				extraLabels: []string{"state"},
				valueType:   prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.TaskStats.NrSleeping),
							labels:    []string{"sleeping"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.TaskStats.NrRunning),
							labels:    []string{"running"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.TaskStats.NrStopped),
							labels:    []string{"stopped"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.TaskStats.NrUninterruptible),
							labels:    []string{"uninterruptible"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.TaskStats.NrIoWait),
							labels:    []string{"iowaiting"},
							timestamp: s.Timestamp,
						},
					}
				},
			},
		}...)
	}
	if includedMetrics.Has(container.HugetlbUsageMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:        "container_hugetlb_failcnt",
				help:        "Number of hugepage usage hits limits",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"pagesize"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.Hugetlb))
					for k, v := range s.Hugetlb {
						values = append(values, metricValue{
							value:     float64(v.Failcnt),
							labels:    []string{k},
							timestamp: s.Timestamp,
						})
					}
					return values
				},
			}, {
				name:        "container_hugetlb_usage_bytes",
				help:        "Current hugepage usage in bytes",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"pagesize"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.Hugetlb))
					for k, v := range s.Hugetlb {
						values = append(values, metricValue{
							value:     float64(v.Usage),
							labels:    []string{k},
							timestamp: s.Timestamp,
						})
					}
					return values
				},
			},
			{
				name:        "container_hugetlb_max_usage_bytes",
				help:        "Maximum hugepage usage recorded in bytes",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"pagesize"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.Hugetlb))
					for k, v := range s.Hugetlb {
						values = append(values, metricValue{
							value:     float64(v.MaxUsage),
							labels:    []string{k},
							timestamp: s.Timestamp,
						})
					}
					return values
				},
			},
		}...)
	}
	if includedMetrics.Has(container.MemoryUsageMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:      "container_memory_cache",
				help:      "Number of bytes of page cache memory.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.Cache), timestamp: s.Timestamp}}
				},
			}, {
				name:      "container_memory_rss",
				help:      "Size of RSS in bytes.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.RSS), timestamp: s.Timestamp}}
				},
			}, {
				name:      "container_memory_kernel_usage",
				help:      "Size of kernel memory allocated in bytes.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.KernelUsage), timestamp: s.Timestamp}}
				},
			}, {
				name:      "container_memory_mapped_file",
				help:      "Size of memory mapped files in bytes.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.MappedFile), timestamp: s.Timestamp}}
				},
			}, {
				name:      "container_memory_swap",
				help:      "Container swap usage in bytes.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.Swap), timestamp: s.Timestamp}}
				},
			}, {
				name:      "container_memory_failcnt",
				help:      "Number of memory usage hits limits",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{
						value:     float64(s.Memory.Failcnt),
						timestamp: s.Timestamp,
					}}
				},
			}, {
				name:      "container_memory_usage_bytes",
				help:      "Current memory usage in bytes, including all memory regardless of when it was accessed",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.Usage), timestamp: s.Timestamp}}
				},
			},
			{
				name:      "container_memory_max_usage_bytes",
				help:      "Maximum memory usage recorded in bytes",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.MaxUsage), timestamp: s.Timestamp}}
				},
			}, {
				name:      "container_memory_working_set_bytes",
				help:      "Current working set in bytes.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.WorkingSet), timestamp: s.Timestamp}}
				},
			},
			{
				name:      "container_memory_total_active_file_bytes",
				help:      "Current total active file in bytes.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.TotalActiveFile), timestamp: s.Timestamp}}
				},
			},
			{
				name:      "container_memory_total_inactive_file_bytes",
				help:      "Current total inactive file in bytes.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.TotalInactiveFile), timestamp: s.Timestamp}}
				},
			},
			{
				name:        "container_memory_failures_total",
				help:        "Cumulative count of memory allocation failures.",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"failure_type", "scope"},
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.Memory.ContainerData.Pgfault),
							labels:    []string{"pgfault", "container"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Memory.ContainerData.Pgmajfault),
							labels:    []string{"pgmajfault", "container"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Memory.HierarchicalData.Pgfault),
							labels:    []string{"pgfault", "hierarchy"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Memory.HierarchicalData.Pgmajfault),
							labels:    []string{"pgmajfault", "hierarchy"},
							timestamp: s.Timestamp,
						},
					}
				},
			},
		}...)
	}
	if includedMetrics.Has(container.CPUSetMetrics) {
		c.containerMetrics = append(c.containerMetrics, containerMetric{
			name:      "container_memory_migrate",
			help:      "Memory migrate status.",
			valueType: prometheus.GaugeValue,
			getValues: func(s *info.ContainerStats) metricValues {
				return metricValues{{value: float64(s.CpuSet.MemoryMigrate), timestamp: s.Timestamp}}
			},
		})
	}
	if includedMetrics.Has(container.MemoryNumaMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:        "container_memory_numa_pages",
				help:        "Number of used pages per NUMA node",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"type", "scope", "node"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0)
					values = append(values, getNumaStatsPerNode(s.Memory.ContainerData.NumaStats.File,
						[]string{"file", "container"}, s.Timestamp)...)
					values = append(values, getNumaStatsPerNode(s.Memory.ContainerData.NumaStats.Anon,
						[]string{"anon", "container"}, s.Timestamp)...)
					values = append(values, getNumaStatsPerNode(s.Memory.ContainerData.NumaStats.Unevictable,
						[]string{"unevictable", "container"}, s.Timestamp)...)

					values = append(values, getNumaStatsPerNode(s.Memory.HierarchicalData.NumaStats.File,
						[]string{"file", "hierarchy"}, s.Timestamp)...)
					values = append(values, getNumaStatsPerNode(s.Memory.HierarchicalData.NumaStats.Anon,
						[]string{"anon", "hierarchy"}, s.Timestamp)...)
					values = append(values, getNumaStatsPerNode(s.Memory.HierarchicalData.NumaStats.Unevictable,
						[]string{"unevictable", "hierarchy"}, s.Timestamp)...)
					return values
				},
			},
		}...)
	}
	if includedMetrics.Has(container.DiskUsageMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:        "container_fs_inodes_free",
				help:        "Number of available Inodes",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.InodesFree)
					}, s.Timestamp)
				},
			}, {
				name:        "container_fs_inodes_total",
				help:        "Number of Inodes",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.Inodes)
					}, s.Timestamp)
				},
			}, {
				name:        "container_fs_limit_bytes",
				help:        "Number of bytes that can be consumed by the container on this filesystem.",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.Limit)
					}, s.Timestamp)
				},
			}, {
				name:        "container_fs_usage_bytes",
				help:        "Number of bytes that are consumed by the container on this filesystem.",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.Usage)
					}, s.Timestamp)
				},
			},
		}...)
	}
	if includedMetrics.Has(container.DiskIOMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:        "container_fs_reads_bytes_total",
				help:        "Cumulative count of bytes read",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.IoServiceBytes, "Read", asFloat64,
						nil, nil,
						s.Timestamp,
					)
				},
			}, {
				name:        "container_fs_reads_total",
				help:        "Cumulative count of reads completed",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.IoServiced, "Read", asFloat64,
						s.Filesystem, func(fs *info.FsStats) float64 {
							return float64(fs.ReadsCompleted)
						},
						s.Timestamp,
					)
				},
			}, {
				name:        "container_fs_sector_reads_total",
				help:        "Cumulative count of sector reads completed",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.Sectors, "Read", asFloat64,
						s.Filesystem, func(fs *info.FsStats) float64 {
							return float64(fs.SectorsRead)
						},
						s.Timestamp,
					)
				},
			}, {
				name:        "container_fs_reads_merged_total",
				help:        "Cumulative count of reads merged",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.IoMerged, "Read", asFloat64,
						s.Filesystem, func(fs *info.FsStats) float64 {
							return float64(fs.ReadsMerged)
						},
						s.Timestamp,
					)
				},
			}, {
				name:        "container_fs_read_seconds_total",
				help:        "Cumulative count of seconds spent reading",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.IoServiceTime, "Read", asNanosecondsToSeconds,
						s.Filesystem, func(fs *info.FsStats) float64 {
							return float64(fs.ReadTime) / float64(time.Second)
						},
						s.Timestamp,
					)
				},
			}, {
				name:        "container_fs_writes_bytes_total",
				help:        "Cumulative count of bytes written",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.IoServiceBytes, "Write", asFloat64,
						nil, nil,
						s.Timestamp,
					)
				},
			}, {
				name:        "container_fs_writes_total",
				help:        "Cumulative count of writes completed",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.IoServiced, "Write", asFloat64,
						s.Filesystem, func(fs *info.FsStats) float64 {
							return float64(fs.WritesCompleted)
						},
						s.Timestamp,
					)
				},
			}, {
				name:        "container_fs_sector_writes_total",
				help:        "Cumulative count of sector writes completed",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.Sectors, "Write", asFloat64,
						s.Filesystem, func(fs *info.FsStats) float64 {
							return float64(fs.SectorsWritten)
						},
						s.Timestamp,
					)
				},
			}, {
				name:        "container_fs_writes_merged_total",
				help:        "Cumulative count of writes merged",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.IoMerged, "Write", asFloat64,
						s.Filesystem, func(fs *info.FsStats) float64 {
							return float64(fs.WritesMerged)
						},
						s.Timestamp,
					)
				},
			}, {
				name:        "container_fs_write_seconds_total",
				help:        "Cumulative count of seconds spent writing",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.IoServiceTime, "Write", asNanosecondsToSeconds,
						s.Filesystem, func(fs *info.FsStats) float64 {
							return float64(fs.WriteTime) / float64(time.Second)
						},
						s.Timestamp,
					)
				},
			}, {
				name:        "container_fs_io_current",
				help:        "Number of I/Os currently in progress",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.IoQueued, "Total", asFloat64,
						s.Filesystem, func(fs *info.FsStats) float64 {
							return float64(fs.IoInProgress)
						},
						s.Timestamp,
					)
				},
			}, {
				name:        "container_fs_io_time_seconds_total",
				help:        "Cumulative count of seconds spent doing I/Os",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.IoServiceTime, "Total", asNanosecondsToSeconds,
						s.Filesystem, func(fs *info.FsStats) float64 {
							return float64(float64(fs.IoTime) / float64(time.Second))
						},
						s.Timestamp,
					)
				},
			}, {
				name:        "container_fs_io_time_weighted_seconds_total",
				help:        "Cumulative weighted I/O time in seconds",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.WeightedIoTime) / float64(time.Second)
					}, s.Timestamp)
				},
			}, {
				name:        "container_fs_io_cost_usage_seconds_total",
				help:        "Cumulative IOCost usage in seconds",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.IoCostUsage, "Count", asMicrosecondsToSeconds,
						[]info.FsStats{}, nil,
						s.Timestamp,
					)
				},
			}, {
				name:        "container_fs_io_cost_wait_seconds_total",
				help:        "Cumulative IOCost wait in seconds",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.IoCostWait, "Count", asMicrosecondsToSeconds,
						[]info.FsStats{}, nil,
						s.Timestamp,
					)
				},
			}, {
				name:        "container_fs_io_cost_indebt_seconds_total",
				help:        "Cumulative IOCost debt in seconds",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.IoCostIndebt, "Count", asMicrosecondsToSeconds,
						[]info.FsStats{}, nil,
						s.Timestamp,
					)
				},
			}, {
				name:        "container_fs_io_cost_indelay_seconds_total",
				help:        "Cumulative IOCost delay in seconds",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.IoCostIndelay, "Count", asMicrosecondsToSeconds,
						[]info.FsStats{}, nil,
						s.Timestamp,
					)
				},
			},
			{
				name:        "container_blkio_device_usage_total",
				help:        "Blkio Device bytes usage",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device", "major", "minor", "operation"},
				getValues: func(s *info.ContainerStats) metricValues {
					var values metricValues
					for _, diskStat := range s.DiskIo.IoServiceBytes {
						for operation, value := range diskStat.Stats {
							values = append(values, metricValue{
								value: float64(value),
								labels: []string{diskStat.Device,
									strconv.Itoa(int(diskStat.Major)),
									strconv.Itoa(int(diskStat.Minor)),
									operation},
								timestamp: s.Timestamp,
							})
						}
					}
					return values
				},
			},
		}...)
	}
	if includedMetrics.Has(container.NetworkUsageMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:        "container_network_receive_bytes_total",
				help:        "Cumulative count of bytes received",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"interface"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.Network.Interfaces))
					for _, value := range s.Network.Interfaces {
						values = append(values, metricValue{
							value:     float64(value.RxBytes),
							labels:    []string{value.Name},
							timestamp: s.Timestamp,
						})
					}
					return values
				},
			}, {
				name:        "container_network_receive_packets_total",
				help:        "Cumulative count of packets received",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"interface"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.Network.Interfaces))
					for _, value := range s.Network.Interfaces {
						values = append(values, metricValue{
							value:     float64(value.RxPackets),
							labels:    []string{value.Name},
							timestamp: s.Timestamp,
						})
					}
					return values
				},
			}, {
				name:        "container_network_receive_packets_dropped_total",
				help:        "Cumulative count of packets dropped while receiving",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"interface"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.Network.Interfaces))
					for _, value := range s.Network.Interfaces {
						values = append(values, metricValue{
							value:     float64(value.RxDropped),
							labels:    []string{value.Name},
							timestamp: s.Timestamp,
						})
					}
					return values
				},
			}, {
				name:        "container_network_receive_errors_total",
				help:        "Cumulative count of errors encountered while receiving",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"interface"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.Network.Interfaces))
					for _, value := range s.Network.Interfaces {
						values = append(values, metricValue{
							value:     float64(value.RxErrors),
							labels:    []string{value.Name},
							timestamp: s.Timestamp,
						})
					}
					return values
				},
			}, {
				name:        "container_network_transmit_bytes_total",
				help:        "Cumulative count of bytes transmitted",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"interface"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.Network.Interfaces))
					for _, value := range s.Network.Interfaces {
						values = append(values, metricValue{
							value:     float64(value.TxBytes),
							labels:    []string{value.Name},
							timestamp: s.Timestamp,
						})
					}
					return values
				},
			}, {
				name:        "container_network_transmit_packets_total",
				help:        "Cumulative count of packets transmitted",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"interface"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.Network.Interfaces))
					for _, value := range s.Network.Interfaces {
						values = append(values, metricValue{
							value:     float64(value.TxPackets),
							labels:    []string{value.Name},
							timestamp: s.Timestamp,
						})
					}
					return values
				},
			}, {
				name:        "container_network_transmit_packets_dropped_total",
				help:        "Cumulative count of packets dropped while transmitting",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"interface"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.Network.Interfaces))
					for _, value := range s.Network.Interfaces {
						values = append(values, metricValue{
							value:     float64(value.TxDropped),
							labels:    []string{value.Name},
							timestamp: s.Timestamp,
						})
					}
					return values
				},
			}, {
				name:        "container_network_transmit_errors_total",
				help:        "Cumulative count of errors encountered while transmitting",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"interface"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.Network.Interfaces))
					for _, value := range s.Network.Interfaces {
						values = append(values, metricValue{
							value:     float64(value.TxErrors),
							labels:    []string{value.Name},
							timestamp: s.Timestamp,
						})
					}
					return values
				},
			},
		}...)
	}
	if includedMetrics.Has(container.NetworkTcpUsageMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:        "container_network_tcp_usage_total",
				help:        "tcp connection usage statistic for container",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"tcp_state"},
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.Network.Tcp.Established),
							labels:    []string{"established"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp.SynSent),
							labels:    []string{"synsent"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp.SynRecv),
							labels:    []string{"synrecv"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp.FinWait1),
							labels:    []string{"finwait1"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp.FinWait2),
							labels:    []string{"finwait2"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp.TimeWait),
							labels:    []string{"timewait"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp.Close),
							labels:    []string{"close"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp.CloseWait),
							labels:    []string{"closewait"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp.LastAck),
							labels:    []string{"lastack"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp.Listen),
							labels:    []string{"listen"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp.Closing),
							labels:    []string{"closing"},
							timestamp: s.Timestamp,
						},
					}
				},
			},
		}...)
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:        "container_network_tcp6_usage_total",
				help:        "tcp6 connection usage statistic for container",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"tcp_state"},
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.Network.Tcp6.Established),
							labels:    []string{"established"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp6.SynSent),
							labels:    []string{"synsent"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp6.SynRecv),
							labels:    []string{"synrecv"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp6.FinWait1),
							labels:    []string{"finwait1"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp6.FinWait2),
							labels:    []string{"finwait2"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp6.TimeWait),
							labels:    []string{"timewait"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp6.Close),
							labels:    []string{"close"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp6.CloseWait),
							labels:    []string{"closewait"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp6.LastAck),
							labels:    []string{"lastack"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp6.Listen),
							labels:    []string{"listen"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Tcp6.Closing),
							labels:    []string{"closing"},
							timestamp: s.Timestamp,
						},
					}
				},
			},
		}...)
	}
	if includedMetrics.Has(container.NetworkAdvancedTcpUsageMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:        "container_network_advance_tcp_stats_total",
				help:        "advance tcp connections statistic for container",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"tcp_state"},
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.Network.TcpAdvanced.RtoAlgorithm),
							labels:    []string{"rtoalgorithm"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.RtoMin),
							labels:    []string{"rtomin"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.RtoMax),
							labels:    []string{"rtomax"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.MaxConn),
							labels:    []string{"maxconn"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.ActiveOpens),
							labels:    []string{"activeopens"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.PassiveOpens),
							labels:    []string{"passiveopens"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.AttemptFails),
							labels:    []string{"attemptfails"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.EstabResets),
							labels:    []string{"estabresets"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.CurrEstab),
							labels:    []string{"currestab"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.InSegs),
							labels:    []string{"insegs"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.OutSegs),
							labels:    []string{"outsegs"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.RetransSegs),
							labels:    []string{"retranssegs"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.InErrs),
							labels:    []string{"inerrs"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.OutRsts),
							labels:    []string{"outrsts"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.InCsumErrors),
							labels:    []string{"incsumerrors"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.EmbryonicRsts),
							labels:    []string{"embryonicrsts"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.SyncookiesSent),
							labels:    []string{"syncookiessent"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.SyncookiesRecv),
							labels:    []string{"syncookiesrecv"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.SyncookiesFailed),
							labels:    []string{"syncookiesfailed"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.PruneCalled),
							labels:    []string{"prunecalled"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.RcvPruned),
							labels:    []string{"rcvpruned"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.OfoPruned),
							labels:    []string{"ofopruned"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.OutOfWindowIcmps),
							labels:    []string{"outofwindowicmps"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.LockDroppedIcmps),
							labels:    []string{"lockdroppedicmps"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TW),
							labels:    []string{"tw"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TWRecycled),
							labels:    []string{"twrecycled"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TWKilled),
							labels:    []string{"twkilled"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPTimeWaitOverflow),
							labels:    []string{"tcptimewaitoverflow"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPTimeouts),
							labels:    []string{"tcptimeouts"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPSpuriousRTOs),
							labels:    []string{"tcpspuriousrtos"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPLossProbes),
							labels:    []string{"tcplossprobes"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPLossProbeRecovery),
							labels:    []string{"tcplossproberecovery"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPRenoRecoveryFail),
							labels:    []string{"tcprenorecoveryfail"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPSackRecoveryFail),
							labels:    []string{"tcpsackrecoveryfail"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPRenoFailures),
							labels:    []string{"tcprenofailures"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPSackFailures),
							labels:    []string{"tcpsackfailures"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPLossFailures),
							labels:    []string{"tcplossfailures"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.DelayedACKs),
							labels:    []string{"delayedacks"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.DelayedACKLocked),
							labels:    []string{"delayedacklocked"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.DelayedACKLost),
							labels:    []string{"delayedacklost"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.ListenOverflows),
							labels:    []string{"listenoverflows"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.ListenDrops),
							labels:    []string{"listendrops"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPHPHits),
							labels:    []string{"tcphphits"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPPureAcks),
							labels:    []string{"tcppureacks"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPHPAcks),
							labels:    []string{"tcphpacks"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPRenoRecovery),
							labels:    []string{"tcprenorecovery"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPSackRecovery),
							labels:    []string{"tcpsackrecovery"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPSACKReneging),
							labels:    []string{"tcpsackreneging"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPFACKReorder),
							labels:    []string{"tcpfackreorder"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPSACKReorder),
							labels:    []string{"tcpsackreorder"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPRenoReorder),
							labels:    []string{"tcprenoreorder"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPTSReorder),
							labels:    []string{"tcptsreorder"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPFullUndo),
							labels:    []string{"tcpfullundo"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPPartialUndo),
							labels:    []string{"tcppartialundo"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPDSACKUndo),
							labels:    []string{"tcpdsackundo"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPLossUndo),
							labels:    []string{"tcplossundo"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPFastRetrans),
							labels:    []string{"tcpfastretrans"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPSlowStartRetrans),
							labels:    []string{"tcpslowstartretrans"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPLostRetransmit),
							labels:    []string{"tcplostretransmit"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPRetransFail),
							labels:    []string{"tcpretransfail"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPRcvCollapsed),
							labels:    []string{"tcprcvcollapsed"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPDSACKOldSent),
							labels:    []string{"tcpdsackoldsent"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPDSACKOfoSent),
							labels:    []string{"tcpdsackofosent"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPDSACKRecv),
							labels:    []string{"tcpdsackrecv"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPDSACKOfoRecv),
							labels:    []string{"tcpdsackoforecv"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPAbortOnData),
							labels:    []string{"tcpabortondata"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPAbortOnClose),
							labels:    []string{"tcpabortonclose"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPAbortOnMemory),
							labels:    []string{"tcpabortonmemory"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPAbortOnTimeout),
							labels:    []string{"tcpabortontimeout"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPAbortOnLinger),
							labels:    []string{"tcpabortonlinger"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPAbortFailed),
							labels:    []string{"tcpabortfailed"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPMemoryPressures),
							labels:    []string{"tcpmemorypressures"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPMemoryPressuresChrono),
							labels:    []string{"tcpmemorypressureschrono"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPSACKDiscard),
							labels:    []string{"tcpsackdiscard"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPDSACKIgnoredOld),
							labels:    []string{"tcpdsackignoredold"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPDSACKIgnoredNoUndo),
							labels:    []string{"tcpdsackignorednoundo"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPMD5NotFound),
							labels:    []string{"tcpmd5notfound"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPMD5Unexpected),
							labels:    []string{"tcpmd5unexpected"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPMD5Failure),
							labels:    []string{"tcpmd5failure"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPSackShifted),
							labels:    []string{"tcpsackshifted"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPSackMerged),
							labels:    []string{"tcpsackmerged"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPSackShiftFallback),
							labels:    []string{"tcpsackshiftfallback"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPBacklogDrop),
							labels:    []string{"tcpbacklogdrop"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.PFMemallocDrop),
							labels:    []string{"pfmemallocdrop"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPMinTTLDrop),
							labels:    []string{"tcpminttldrop"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPDeferAcceptDrop),
							labels:    []string{"tcpdeferacceptdrop"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.IPReversePathFilter),
							labels:    []string{"ipreversepathfilter"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPReqQFullDoCookies),
							labels:    []string{"tcpreqqfulldocookies"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPReqQFullDrop),
							labels:    []string{"tcpreqqfulldrop"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPFastOpenActive),
							labels:    []string{"tcpfastopenactive"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPFastOpenActiveFail),
							labels:    []string{"tcpfastopenactivefail"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPFastOpenPassive),
							labels:    []string{"tcpfastopenpassive"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPFastOpenPassiveFail),
							labels:    []string{"tcpfastopenpassivefail"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPFastOpenListenOverflow),
							labels:    []string{"tcpfastopenlistenoverflow"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPFastOpenCookieReqd),
							labels:    []string{"tcpfastopencookiereqd"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPSynRetrans),
							labels:    []string{"tcpsynretrans"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.TCPOrigDataSent),
							labels:    []string{"tcporigdatasent"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.PAWSActive),
							labels:    []string{"pawsactive"},
							timestamp: s.Timestamp,
						}, {
							value:     float64(s.Network.TcpAdvanced.PAWSEstab),
							labels:    []string{"pawsestab"},
							timestamp: s.Timestamp,
						},
					}
				},
			},
		}...)
	}
	if includedMetrics.Has(container.NetworkUdpUsageMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:        "container_network_udp6_usage_total",
				help:        "udp6 connection usage statistic for container",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"udp_state"},
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.Network.Udp6.Listen),
							labels:    []string{"listen"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Udp6.Dropped),
							labels:    []string{"dropped"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Udp6.RxQueued),
							labels:    []string{"rxqueued"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Udp6.TxQueued),
							labels:    []string{"txqueued"},
							timestamp: s.Timestamp,
						},
					}
				},
			},
		}...)
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:        "container_network_udp_usage_total",
				help:        "udp connection usage statistic for container",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"udp_state"},
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.Network.Udp.Listen),
							labels:    []string{"listen"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Udp.Dropped),
							labels:    []string{"dropped"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Udp.RxQueued),
							labels:    []string{"rxqueued"},
							timestamp: s.Timestamp,
						},
						{
							value:     float64(s.Network.Udp.TxQueued),
							labels:    []string{"txqueued"},
							timestamp: s.Timestamp,
						},
					}
				},
			},
		}...)
	}
	if includedMetrics.Has(container.ProcessMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:      "container_processes",
				help:      "Number of processes running inside the container.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Processes.ProcessCount), timestamp: s.Timestamp}}
				},
			},
			{
				name:      "container_file_descriptors",
				help:      "Number of open file descriptors for the container.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Processes.FdCount), timestamp: s.Timestamp}}
				},
			},
			{
				name:      "container_sockets",
				help:      "Number of open sockets for the container.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Processes.SocketCount), timestamp: s.Timestamp}}
				},
			},
			{
				name:      "container_threads_max",
				help:      "Maximum number of threads allowed inside the container, infinity if value is zero",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.Processes.ThreadsMax),
							timestamp: s.Timestamp,
						},
					}
				},
			},
			{
				name:      "container_threads",
				help:      "Number of threads running inside the container",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:     float64(s.Processes.ThreadsCurrent),
							timestamp: s.Timestamp,
						},
					}
				},
			},
			{
				name:        "container_ulimits_soft",
				help:        "Soft ulimit values for the container root process. Unlimited if -1, except priority and nice",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"ulimit"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.Processes.Ulimits))
					for _, ulimit := range s.Processes.Ulimits {
						values = append(values, metricValue{
							value:     float64(ulimit.SoftLimit),
							labels:    []string{ulimit.Name},
							timestamp: s.Timestamp,
						})
					}
					return values
				},
			},
		}...)
	}
	if includedMetrics.Has(container.PerfMetrics) {
		if includedMetrics.Has(container.PerCpuUsageMetrics) {
			c.containerMetrics = append(c.containerMetrics, []containerMetric{
				{
					name:        "container_perf_events_total",
					help:        "Perf event metric.",
					valueType:   prometheus.CounterValue,
					extraLabels: []string{"cpu", "event"},
					getValues: func(s *info.ContainerStats) metricValues {
						return getPerCPUCorePerfEvents(s)
					},
				},
				{
					name:        "container_perf_events_scaling_ratio",
					help:        "Perf event metric scaling ratio.",
					valueType:   prometheus.GaugeValue,
					extraLabels: []string{"cpu", "event"},
					getValues: func(s *info.ContainerStats) metricValues {
						return getPerCPUCoreScalingRatio(s)
					},
				}}...)
		} else {
			c.containerMetrics = append(c.containerMetrics, []containerMetric{
				{
					name:        "container_perf_events_total",
					help:        "Perf event metric.",
					valueType:   prometheus.CounterValue,
					extraLabels: []string{"cpu", "event"},
					getValues: func(s *info.ContainerStats) metricValues {
						return getAggregatedCorePerfEvents(s)
					},
				},
				{
					name:        "container_perf_events_scaling_ratio",
					help:        "Perf event metric scaling ratio.",
					valueType:   prometheus.GaugeValue,
					extraLabels: []string{"cpu", "event"},
					getValues: func(s *info.ContainerStats) metricValues {
						return getMinCoreScalingRatio(s)
					},
				}}...)
		}
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:        "container_perf_uncore_events_total",
				help:        "Perf uncore event metric.",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"socket", "event", "pmu"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.PerfUncoreStats))
					for _, metric := range s.PerfUncoreStats {
						values = append(values, metricValue{
							value:     float64(metric.Value),
							labels:    []string{strconv.Itoa(metric.Socket), metric.Name, metric.PMU},
							timestamp: s.Timestamp,
						})
					}
					return values
				},
			},
			{
				name:        "container_perf_uncore_events_scaling_ratio",
				help:        "Perf uncore event metric scaling ratio.",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"socket", "event", "pmu"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.PerfUncoreStats))
					for _, metric := range s.PerfUncoreStats {
						values = append(values, metricValue{
							value:     metric.ScalingRatio,
							labels:    []string{strconv.Itoa(metric.Socket), metric.Name, metric.PMU},
							timestamp: s.Timestamp,
						})
					}
					return values
				},
			},
		}...)
	}
	if includedMetrics.Has(container.ReferencedMemoryMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:      "container_referenced_bytes",
				help:      "Container referenced bytes during last measurements cycle",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.ReferencedMemory), timestamp: s.Timestamp}}
				},
			},
		}...)
	}
	if includedMetrics.Has(container.ResctrlMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:        "container_memory_bandwidth_bytes",
				help:        "Total memory bandwidth usage statistics for container counted with RDT Memory Bandwidth Monitoring (MBM).",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{prometheusNodeLabelName},
				getValues: func(s *info.ContainerStats) metricValues {
					numberOfNUMANodes := len(s.Resctrl.MemoryBandwidth)
					metrics := make(metricValues, numberOfNUMANodes)
					for numaNode, stats := range s.Resctrl.MemoryBandwidth {
						metrics[numaNode] = metricValue{
							value:     float64(stats.TotalBytes),
							timestamp: s.Timestamp,
							labels:    []string{strconv.Itoa(numaNode)},
						}
					}
					return metrics
				},
			},
			{
				name:        "container_memory_bandwidth_local_bytes",
				help:        "Local memory bandwidth usage statistics for container counted with RDT Memory Bandwidth Monitoring (MBM).",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{prometheusNodeLabelName},
				getValues: func(s *info.ContainerStats) metricValues {
					numberOfNUMANodes := len(s.Resctrl.MemoryBandwidth)
					metrics := make(metricValues, numberOfNUMANodes)
					for numaNode, stats := range s.Resctrl.MemoryBandwidth {
						metrics[numaNode] = metricValue{
							value:     float64(stats.LocalBytes),
							timestamp: s.Timestamp,
							labels:    []string{strconv.Itoa(numaNode)},
						}
					}
					return metrics
				},
			},
			{
				name:        "container_llc_occupancy_bytes",
				help:        "Last level cache usage statistics for container counted with RDT Memory Bandwidth Monitoring (MBM).",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{prometheusNodeLabelName},
				getValues: func(s *info.ContainerStats) metricValues {
					numberOfNUMANodes := len(s.Resctrl.Cache)
					metrics := make(metricValues, numberOfNUMANodes)
					for numaNode, stats := range s.Resctrl.Cache {
						metrics[numaNode] = metricValue{
							value:     float64(stats.LLCOccupancy),
							timestamp: s.Timestamp,
							labels:    []string{strconv.Itoa(numaNode)},
						}
					}
					return metrics
				},
			},
		}...)
	}
	if includedMetrics.Has(container.OOMMetrics) {
		c.containerMetrics = append(c.containerMetrics, containerMetric{
			name:      "container_oom_events_total",
			help:      "Count of out of memory events observed for the container",
			valueType: prometheus.CounterValue,
			getValues: func(s *info.ContainerStats) metricValues {
				return metricValues{{value: float64(s.OOMEvents), timestamp: s.Timestamp}}
			},
		})
	}

	if includedMetrics.Has(container.PressureMetrics) {
		c.containerMetrics = append(c.containerMetrics, []containerMetric{
			{
				name:      "container_pressure_cpu_stalled_seconds_total",
				help:      "Total time duration no tasks in the container could make progress due to CPU congestion.",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: asMicrosecondsToSeconds(s.Cpu.PSI.Full.Total), timestamp: s.Timestamp}}
				},
			}, {
				name:      "container_pressure_cpu_waiting_seconds_total",
				help:      "Total time duration tasks in the container have waited due to CPU congestion.",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: asMicrosecondsToSeconds(s.Cpu.PSI.Some.Total), timestamp: s.Timestamp}}
				},
			}, {
				name:      "container_pressure_memory_stalled_seconds_total",
				help:      "Total time duration no tasks in the container could make progress due to memory congestion.",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: asMicrosecondsToSeconds(s.Memory.PSI.Full.Total), timestamp: s.Timestamp}}
				},
			}, {
				name:      "container_pressure_memory_waiting_seconds_total",
				help:      "Total time duration tasks in the container have waited due to memory congestion.",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: asMicrosecondsToSeconds(s.Memory.PSI.Some.Total), timestamp: s.Timestamp}}
				},
			}, {
				name:      "container_pressure_io_stalled_seconds_total",
				help:      "Total time duration no tasks in the container could make progress due to IO congestion.",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: asMicrosecondsToSeconds(s.DiskIo.PSI.Full.Total), timestamp: s.Timestamp}}
				},
			}, {
				name:      "container_pressure_io_waiting_seconds_total",
				help:      "Total time duration tasks in the container have waited due to IO congestion.",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: asMicrosecondsToSeconds(s.DiskIo.PSI.Some.Total), timestamp: s.Timestamp}}
				},
			},
		}...)
	}

	return c
}

var (
	versionInfoDesc = prometheus.NewDesc("cadvisor_version_info", "A metric with a constant '1' value labeled by kernel version, OS version, docker version, cadvisor version & cadvisor revision.", []string{"kernelVersion", "osVersion", "dockerVersion", "cadvisorVersion", "cadvisorRevision"}, nil)
	startTimeDesc   = prometheus.NewDesc("container_start_time_seconds", "Start time of the container since unix epoch in seconds.", nil, nil)
	cpuPeriodDesc   = prometheus.NewDesc("container_spec_cpu_period", "CPU period of the container.", nil, nil)
	cpuQuotaDesc    = prometheus.NewDesc("container_spec_cpu_quota", "CPU quota of the container.", nil, nil)
	cpuSharesDesc   = prometheus.NewDesc("container_spec_cpu_shares", "CPU share of the container.", nil, nil)
)

// Describe describes all the metrics ever exported by cadvisor. It
// implements prometheus.PrometheusCollector.
func (c *PrometheusCollector) Describe(ch chan<- *prometheus.Desc) {
	c.errors.Describe(ch)
	for _, cm := range c.containerMetrics {
		ch <- cm.desc([]string{})
	}
	ch <- startTimeDesc
	ch <- cpuPeriodDesc
	ch <- cpuQuotaDesc
	ch <- cpuSharesDesc
	ch <- versionInfoDesc
}

// Collect fetches the stats from all containers and delivers them as
// Prometheus metrics. It implements prometheus.PrometheusCollector.
func (c *PrometheusCollector) Collect(ch chan<- prometheus.Metric) {
	c.errors.Set(0)
	c.collectVersionInfo(ch)
	c.collectContainersInfo(ch)
	c.errors.Collect(ch)
}

const (
	// ContainerLabelPrefix is the prefix added to all container labels.
	ContainerLabelPrefix = "container_label_"
	// ContainerEnvPrefix is the prefix added to all env variable labels.
	ContainerEnvPrefix = "container_env_"
	// LabelID is the name of the id label.
	LabelID = "id"
	// LabelName is the name of the name label.
	LabelName = "name"
	// LabelImage is the name of the image label.
	LabelImage = "image"
)

// DefaultContainerLabels implements ContainerLabelsFunc. It exports the
// container name, first alias, image name as well as all its env and label
// values.
func DefaultContainerLabels(container *info.ContainerInfo) map[string]string {
	set := map[string]string{LabelID: container.Name}
	if len(container.Aliases) > 0 {
		set[LabelName] = container.Aliases[0]
	}
	if image := container.Spec.Image; len(image) > 0 {
		set[LabelImage] = image
	}
	for k, v := range container.Spec.Labels {
		set[ContainerLabelPrefix+k] = v
	}
	for k, v := range container.Spec.Envs {
		set[ContainerEnvPrefix+k] = v
	}
	return set
}

// BaseContainerLabels returns a ContainerLabelsFunc that exports the container
// name, first alias, image name as well as all its white listed env and label values.
func BaseContainerLabels(whiteList []string) func(container *info.ContainerInfo) map[string]string {
	whiteListMap := make(map[string]struct{}, len(whiteList))
	for _, k := range whiteList {
		whiteListMap[k] = struct{}{}
	}

	return func(container *info.ContainerInfo) map[string]string {
		set := map[string]string{LabelID: container.Name}
		if len(container.Aliases) > 0 {
			set[LabelName] = container.Aliases[0]
		}
		if image := container.Spec.Image; len(image) > 0 {
			set[LabelImage] = image
		}
		for k, v := range container.Spec.Labels {
			if _, ok := whiteListMap[k]; ok {
				set[ContainerLabelPrefix+k] = v
			}
		}
		for k, v := range container.Spec.Envs {
			set[ContainerEnvPrefix+k] = v
		}
		return set
	}
}

func (c *PrometheusCollector) collectContainersInfo(ch chan<- prometheus.Metric) {
	containers, err := c.infoProvider.GetRequestedContainersInfo("/", c.opts)
	if err != nil {
		c.errors.Set(1)
		klog.Warningf("Couldn't get containers: %s", err)
		return
	}
	rawLabels := map[string]struct{}{}
	for _, container := range containers {
		for l := range c.containerLabelsFunc(container) {
			rawLabels[l] = struct{}{}
		}
	}

	for _, cont := range containers {
		values := make([]string, 0, len(rawLabels))
		labels := make([]string, 0, len(rawLabels))
		containerLabels := c.containerLabelsFunc(cont)
		for l := range rawLabels {
			duplicate := false
			sl := sanitizeLabelName(l)
			for _, x := range labels {
				if sl == x {
					duplicate = true
					break
				}
			}
			if !duplicate {
				labels = append(labels, sl)
				values = append(values, containerLabels[l])
			}
		}

		// Container spec
		desc := prometheus.NewDesc("container_start_time_seconds", "Start time of the container since unix epoch in seconds.", labels, nil)
		ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, float64(cont.Spec.CreationTime.Unix()), values...)

		if cont.Spec.HasCpu {
			desc = prometheus.NewDesc("container_spec_cpu_period", "CPU period of the container.", labels, nil)
			ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, float64(cont.Spec.Cpu.Period), values...)
			if cont.Spec.Cpu.Quota != 0 {
				desc = prometheus.NewDesc("container_spec_cpu_quota", "CPU quota of the container.", labels, nil)
				ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, float64(cont.Spec.Cpu.Quota), values...)
			}
			desc := prometheus.NewDesc("container_spec_cpu_shares", "CPU share of the container.", labels, nil)
			ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, float64(cont.Spec.Cpu.Limit), values...)

		}
		if cont.Spec.HasMemory {
			desc := prometheus.NewDesc("container_spec_memory_limit_bytes", "Memory limit for the container.", labels, nil)
			ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, specMemoryValue(cont.Spec.Memory.Limit), values...)
			desc = prometheus.NewDesc("container_spec_memory_swap_limit_bytes", "Memory swap limit for the container.", labels, nil)
			ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, specMemoryValue(cont.Spec.Memory.SwapLimit), values...)
			desc = prometheus.NewDesc("container_spec_memory_reservation_limit_bytes", "Memory reservation limit for the container.", labels, nil)
			ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, specMemoryValue(cont.Spec.Memory.Reservation), values...)
		}

		// Now for the actual metrics
		if len(cont.Stats) == 0 {
			continue
		}
		stats := cont.Stats[0]
		for _, cm := range c.containerMetrics {
			if cm.condition != nil && !cm.condition(cont.Spec) {
				continue
			}
			desc := cm.desc(labels)
			for _, metricValue := range cm.getValues(stats) {
				ch <- prometheus.NewMetricWithTimestamp(
					metricValue.timestamp,
					prometheus.MustNewConstMetric(desc, cm.valueType, float64(metricValue.value), append(values, metricValue.labels...)...),
				)
			}
		}
		if c.includedMetrics.Has(container.AppMetrics) {
			for metricLabel, v := range stats.CustomMetrics {
				for _, metric := range v {
					clabels := make([]string, len(rawLabels), len(rawLabels)+len(metric.Labels))
					cvalues := make([]string, len(rawLabels), len(rawLabels)+len(metric.Labels))
					copy(clabels, labels)
					copy(cvalues, values)
					for label, value := range metric.Labels {
						clabels = append(clabels, sanitizeLabelName("app_"+label))
						cvalues = append(cvalues, value)
					}
					desc := prometheus.NewDesc(metricLabel, "Custom application metric.", clabels, nil)
					ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, float64(metric.FloatValue), cvalues...)
				}
			}
		}
	}
}

func (c *PrometheusCollector) collectVersionInfo(ch chan<- prometheus.Metric) {
	versionInfo, err := c.infoProvider.GetVersionInfo()
	if err != nil {
		c.errors.Set(1)
		klog.Warningf("Couldn't get version info: %s", err)
		return
	}
	ch <- prometheus.MustNewConstMetric(versionInfoDesc, prometheus.GaugeValue, 1, []string{versionInfo.KernelVersion, versionInfo.ContainerOsVersion, versionInfo.DockerVersion, versionInfo.CadvisorVersion, versionInfo.CadvisorRevision}...)
}

// Size after which we consider memory to be "unlimited". This is not
// MaxInt64 due to rounding by the kernel.
const maxMemorySize = uint64(1 << 62)

func specMemoryValue(v uint64) float64 {
	if v > maxMemorySize {
		return 0
	}
	return float64(v)
}

var invalidNameCharRE = regexp.MustCompile(`[^a-zA-Z0-9_]`)

// sanitizeLabelName replaces anything that doesn't match
// client_label.LabelNameRE with an underscore.
func sanitizeLabelName(name string) string {
	return invalidNameCharRE.ReplaceAllString(name, "_")
}

func getNumaStatsPerNode(nodeStats map[uint8]uint64, labels []string, timestamp time.Time) metricValues {
	mValues := make(metricValues, 0, len(nodeStats))
	for node, stat := range nodeStats {
		nodeLabels := append(labels, strconv.FormatUint(uint64(node), 10))
		mValues = append(mValues, metricValue{value: float64(stat), labels: nodeLabels, timestamp: timestamp})
	}
	return mValues
}

func getPerCPUCorePerfEvents(s *info.ContainerStats) metricValues {
	values := make(metricValues, 0, len(s.PerfStats))
	for _, metric := range s.PerfStats {
		values = append(values, metricValue{
			value:     float64(metric.Value),
			labels:    []string{strconv.Itoa(metric.Cpu), metric.Name},
			timestamp: s.Timestamp,
		})
	}
	return values
}

func getPerCPUCoreScalingRatio(s *info.ContainerStats) metricValues {
	values := make(metricValues, 0, len(s.PerfStats))
	for _, metric := range s.PerfStats {
		values = append(values, metricValue{
			value:     metric.ScalingRatio,
			labels:    []string{strconv.Itoa(metric.Cpu), metric.Name},
			timestamp: s.Timestamp,
		})
	}
	return values
}

func getAggregatedCorePerfEvents(s *info.ContainerStats) metricValues {
	values := make(metricValues, 0)

	perfEventStatAgg := make(map[string]uint64)
	// aggregate by event
	for _, perfStat := range s.PerfStats {
		perfEventStatAgg[perfStat.Name] += perfStat.Value
	}
	// create aggregated metrics
	for perfEvent, perfValue := range perfEventStatAgg {
		values = append(values, metricValue{
			value:     float64(perfValue),
			labels:    []string{"", perfEvent},
			timestamp: s.Timestamp,
		})
	}
	return values
}

func getMinCoreScalingRatio(s *info.ContainerStats) metricValues {
	values := make(metricValues, 0)
	perfEventStatMin := make(map[string]float64)
	// search for minimal value of scalin ratio for specific event
	for _, perfStat := range s.PerfStats {
		if _, ok := perfEventStatMin[perfStat.Name]; !ok {
			// found a new event
			perfEventStatMin[perfStat.Name] = perfStat.ScalingRatio
		} else if perfStat.ScalingRatio < perfEventStatMin[perfStat.Name] {
			// found a lower value of scaling ration so replace the minimal value
			perfEventStatMin[perfStat.Name] = perfStat.ScalingRatio
		}
	}

	for perfEvent, perfScalingRatio := range perfEventStatMin {
		values = append(values, metricValue{
			value:     perfScalingRatio,
			labels:    []string{"", perfEvent},
			timestamp: s.Timestamp,
		})
	}
	return values
}

func getContainerHealthState(s *info.ContainerStats) metricValues {
	value := float64(0)
	switch s.Health.Status {
	case "healthy":
		value = 1
	case "": // if container has no health check defined
		value = -1
	default: // starting or unhealthy
	}
	return metricValues{{
		value:     value,
		timestamp: s.Timestamp,
	}}
}
