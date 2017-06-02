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
	"time"

	info "github.com/google/cadvisor/info/v1"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
)

// infoProvider will usually be manager.Manager, but can be swapped out for testing.
type infoProvider interface {
	// SubcontainersInfo provides information about all subcontainers of the
	// specified container including itself.
	SubcontainersInfo(containerName string, query *info.ContainerInfoRequest) ([]*info.ContainerInfo, error)
	// GetVersionInfo provides information about the version.
	GetVersionInfo() (*info.VersionInfo, error)
	// GetMachineInfo provides information about the machine.
	GetMachineInfo() (*info.MachineInfo, error)
}

// metricValue describes a single metric value for a given set of label values
// within a parent containerMetric.
type metricValue struct {
	value  float64
	labels []string
}

type metricValues []metricValue

// asFloat64 converts a uint64 into a float64.
func asFloat64(v uint64) float64 { return float64(v) }

// asNanosecondsToSeconds converts nanoseconds into a float64 representing seconds.
func asNanosecondsToSeconds(v uint64) float64 {
	return float64(v) / float64(time.Second)
}

// fsValues is a helper method for assembling per-filesystem stats.
func fsValues(fsStats []info.FsStats, valueFn func(*info.FsStats) float64) metricValues {
	values := make(metricValues, 0, len(fsStats))
	for _, stat := range fsStats {
		values = append(values, metricValue{
			value:  valueFn(&stat),
			labels: []string{stat.Device},
		})
	}
	return values
}

// ioValues is a helper method for assembling per-disk and per-filesystem stats.
func ioValues(ioStats []info.PerDiskStats, ioType string, ioValueFn func(uint64) float64, fsStats []info.FsStats, valueFn func(*info.FsStats) float64) metricValues {
	values := make(metricValues, 0, len(ioStats)+len(fsStats))
	for _, stat := range ioStats {
		values = append(values, metricValue{
			value:  ioValueFn(stat.Stats[ioType]),
			labels: []string{stat.Device},
		})
	}
	for _, stat := range fsStats {
		values = append(values, metricValue{
			value:  valueFn(&stat),
			labels: []string{stat.Device},
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
}

// NewPrometheusCollector returns a new PrometheusCollector. The passed
// ContainerLabelsFunc specifies which base labels will be attached to all
// exported metrics. If left to nil, the DefaultContainerLabels function
// will be used instead.
func NewPrometheusCollector(i infoProvider, f ContainerLabelsFunc) *PrometheusCollector {
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
					return metricValues{{value: float64(time.Now().Unix())}}
				},
			}, {
				name:      "container_cpu_user_seconds_total",
				help:      "Cumulative user cpu time consumed in seconds.",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Cpu.Usage.User) / float64(time.Second)}}
				},
			}, {
				name:      "container_cpu_system_seconds_total",
				help:      "Cumulative system cpu time consumed in seconds.",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Cpu.Usage.System) / float64(time.Second)}}
				},
			}, {
				name:        "container_cpu_usage_seconds_total",
				help:        "Cumulative cpu time consumed per cpu in seconds.",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"cpu"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.Cpu.Usage.PerCpu))
					for i, value := range s.Cpu.Usage.PerCpu {
						if value > 0 {
							values = append(values, metricValue{
								value:  float64(value) / float64(time.Second),
								labels: []string{fmt.Sprintf("cpu%02d", i)},
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
					return metricValues{{value: float64(s.Cpu.CFS.Periods)}}
				},
			}, {
				name:      "container_cpu_cfs_throttled_periods_total",
				help:      "Number of throttled period intervals.",
				valueType: prometheus.CounterValue,
				condition: func(s info.ContainerSpec) bool { return s.Cpu.Quota != 0 },
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Cpu.CFS.ThrottledPeriods)}}
				},
			}, {
				name:      "container_cpu_cfs_throttled_seconds_total",
				help:      "Total time duration the container has been throttled.",
				valueType: prometheus.CounterValue,
				condition: func(s info.ContainerSpec) bool { return s.Cpu.Quota != 0 },
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Cpu.CFS.ThrottledTime) / float64(time.Second)}}
				},
			}, {
				name:      "container_memory_cache",
				help:      "Number of bytes of page cache memory.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.Cache)}}
				},
			}, {
				name:      "container_memory_rss",
				help:      "Size of RSS in bytes.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.RSS)}}
				},
			}, {
				name:      "container_memory_swap",
				help:      "Container swap usage in bytes.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.Swap)}}
				},
			}, {
				name:      "container_memory_failcnt",
				help:      "Number of memory usage hits limits",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.Failcnt)}}
				},
			}, {
				name:      "container_memory_usage_bytes",
				help:      "Current memory usage in bytes.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.Usage)}}
				},
			}, {
				name:      "container_memory_working_set_bytes",
				help:      "Current working set in bytes.",
				valueType: prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Memory.WorkingSet)}}
				},
			}, {
				name:        "container_memory_failures_total",
				help:        "Cumulative count of memory allocation failures.",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"type", "scope"},
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:  float64(s.Memory.ContainerData.Pgfault),
							labels: []string{"pgfault", "container"},
						},
						{
							value:  float64(s.Memory.ContainerData.Pgmajfault),
							labels: []string{"pgmajfault", "container"},
						},
						{
							value:  float64(s.Memory.HierarchicalData.Pgfault),
							labels: []string{"pgfault", "hierarchy"},
						},
						{
							value:  float64(s.Memory.HierarchicalData.Pgmajfault),
							labels: []string{"pgmajfault", "hierarchy"},
						},
					}
				},
			}, {
				name:        "container_fs_inodes_free",
				help:        "Number of available Inodes",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.InodesFree)
					})
				},
			}, {
				name:        "container_fs_inodes_total",
				help:        "Number of Inodes",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.Inodes)
					})
				},
			}, {
				name:        "container_fs_limit_bytes",
				help:        "Number of bytes that can be consumed by the container on this filesystem.",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.Limit)
					})
				},
			}, {
				name:        "container_fs_usage_bytes",
				help:        "Number of bytes that are consumed by the container on this filesystem.",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.Usage)
					})
				},
			}, {
				name:        "container_fs_reads_bytes_total",
				help:        "Cumulative count of bytes read",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return ioValues(
						s.DiskIo.IoServiceBytes, "Read", asFloat64,
						nil, nil,
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
					})
				},
			}, {
				name:        "container_network_receive_bytes_total",
				help:        "Cumulative count of bytes received",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"interface"},
				getValues: func(s *info.ContainerStats) metricValues {
					values := make(metricValues, 0, len(s.Network.Interfaces))
					for _, value := range s.Network.Interfaces {
						values = append(values, metricValue{
							value:  float64(value.RxBytes),
							labels: []string{value.Name},
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
							value:  float64(value.RxPackets),
							labels: []string{value.Name},
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
							value:  float64(value.RxDropped),
							labels: []string{value.Name},
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
							value:  float64(value.RxErrors),
							labels: []string{value.Name},
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
							value:  float64(value.TxBytes),
							labels: []string{value.Name},
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
							value:  float64(value.TxPackets),
							labels: []string{value.Name},
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
							value:  float64(value.TxDropped),
							labels: []string{value.Name},
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
							value:  float64(value.TxErrors),
							labels: []string{value.Name},
						})
					}
					return values
				},
			}, {
				name:        "container_tasks_state",
				help:        "Number of tasks in given state",
				extraLabels: []string{"state"},
				valueType:   prometheus.GaugeValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{
						{
							value:  float64(s.TaskStats.NrSleeping),
							labels: []string{"sleeping"},
						},
						{
							value:  float64(s.TaskStats.NrRunning),
							labels: []string{"running"},
						},
						{
							value:  float64(s.TaskStats.NrStopped),
							labels: []string{"stopped"},
						},
						{
							value:  float64(s.TaskStats.NrUninterruptible),
							labels: []string{"uninterruptible"},
						},
						{
							value:  float64(s.TaskStats.NrIoWait),
							labels: []string{"iowaiting"},
						},
					}
				},
			},
		},
	}
	return c
}

var (
	versionInfoDesc       = prometheus.NewDesc("cadvisor_version_info", "A metric with a constant '1' value labeled by kernel version, OS version, docker version, cadvisor version & cadvisor revision.", []string{"kernelVersion", "osVersion", "dockerVersion", "cadvisorVersion", "cadvisorRevision"}, nil)
	machineInfoCoresDesc  = prometheus.NewDesc("machine_cpu_cores", "Number of CPU cores on the machine.", nil, nil)
	machineInfoMemoryDesc = prometheus.NewDesc("machine_memory_bytes", "Amount of memory installed on the machine.", nil, nil)
)

// Describe describes all the metrics ever exported by cadvisor. It
// implements prometheus.PrometheusCollector.
func (c *PrometheusCollector) Describe(ch chan<- *prometheus.Desc) {
	c.errors.Describe(ch)
	for _, cm := range c.containerMetrics {
		ch <- cm.desc([]string{})
	}
	ch <- versionInfoDesc
	ch <- machineInfoCoresDesc
	ch <- machineInfoMemoryDesc
}

// Collect fetches the stats from all containers and delivers them as
// Prometheus metrics. It implements prometheus.PrometheusCollector.
func (c *PrometheusCollector) Collect(ch chan<- prometheus.Metric) {
	c.errors.Set(0)
	c.collectMachineInfo(ch)
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

func (c *PrometheusCollector) collectContainersInfo(ch chan<- prometheus.Metric) {
	containers, err := c.infoProvider.SubcontainersInfo("/", &info.ContainerInfoRequest{NumStats: 1})
	if err != nil {
		c.errors.Set(1)
		glog.Warningf("Couldn't get containers: %s", err)
		return
	}
	for _, container := range containers {
		labels, values := []string{}, []string{}
		for l, v := range c.containerLabelsFunc(container) {
			labels = append(labels, sanitizeLabelName(l))
			values = append(values, v)
		}

		// Container spec
		desc := prometheus.NewDesc("container_start_time_seconds", "Start time of the container since unix epoch in seconds.", labels, nil)
		ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, float64(container.Spec.CreationTime.Unix()), values...)

		if container.Spec.HasCpu {
			desc = prometheus.NewDesc("container_spec_cpu_period", "CPU period of the container.", labels, nil)
			ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, float64(container.Spec.Cpu.Period), values...)
			if container.Spec.Cpu.Quota != 0 {
				desc = prometheus.NewDesc("container_spec_cpu_quota", "CPU quota of the container.", labels, nil)
				ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, float64(container.Spec.Cpu.Quota), values...)
			}
			desc := prometheus.NewDesc("container_spec_cpu_shares", "CPU share of the container.", labels, nil)
			ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, float64(container.Spec.Cpu.Limit), values...)

		}
		if container.Spec.HasMemory {
			desc := prometheus.NewDesc("container_spec_memory_limit_bytes", "Memory limit for the container.", labels, nil)
			ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, specMemoryValue(container.Spec.Memory.Limit), values...)
			desc = prometheus.NewDesc("container_spec_memory_swap_limit_bytes", "Memory swap limit for the container.", labels, nil)
			ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, specMemoryValue(container.Spec.Memory.SwapLimit), values...)
		}

		// Now for the actual metrics
		stats := container.Stats[0]
		for _, cm := range c.containerMetrics {
			if cm.condition != nil && !cm.condition(container.Spec) {
				continue
			}
			desc := cm.desc(labels)
			for _, metricValue := range cm.getValues(stats) {
				ch <- prometheus.MustNewConstMetric(desc, cm.valueType, float64(metricValue.value), append(values, metricValue.labels...)...)
			}
		}
	}
}

func (c *PrometheusCollector) collectVersionInfo(ch chan<- prometheus.Metric) {
	versionInfo, err := c.infoProvider.GetVersionInfo()
	if err != nil {
		c.errors.Set(1)
		glog.Warningf("Couldn't get version info: %s", err)
		return
	}
	ch <- prometheus.MustNewConstMetric(versionInfoDesc, prometheus.GaugeValue, 1, []string{versionInfo.KernelVersion, versionInfo.ContainerOsVersion, versionInfo.DockerVersion, versionInfo.CadvisorVersion, versionInfo.CadvisorRevision}...)
}

func (c *PrometheusCollector) collectMachineInfo(ch chan<- prometheus.Metric) {
	machineInfo, err := c.infoProvider.GetMachineInfo()
	if err != nil {
		c.errors.Set(1)
		glog.Warningf("Couldn't get machine info: %s", err)
		return
	}
	ch <- prometheus.MustNewConstMetric(machineInfoCoresDesc, prometheus.GaugeValue, float64(machineInfo.NumCores))
	ch <- prometheus.MustNewConstMetric(machineInfoMemoryDesc, prometheus.GaugeValue, float64(machineInfo.MemoryCapacity))
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

var invalidLabelCharRE = regexp.MustCompile(`[^a-zA-Z0-9_]`)

// sanitizeLabelName replaces anything that doesn't match
// client_label.LabelNameRE with an underscore.
func sanitizeLabelName(name string) string {
	return invalidLabelCharRE.ReplaceAllString(name, "_")
}
