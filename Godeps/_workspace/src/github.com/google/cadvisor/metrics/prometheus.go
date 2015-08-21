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
	"time"

	"github.com/golang/glog"
	info "github.com/google/cadvisor/info/v1"
	"github.com/prometheus/client_golang/prometheus"
)

// This will usually be manager.Manager, but can be swapped out for testing.
type subcontainersInfoProvider interface {
	// Get information about all subcontainers of the specified container (includes self).
	SubcontainersInfo(containerName string, query *info.ContainerInfoRequest) ([]*info.ContainerInfo, error)
}

// metricValue describes a single metric value for a given set of label values
// within a parent containerMetric.
type metricValue struct {
	value  float64
	labels []string
}

type metricValues []metricValue

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

// A containerMetric describes a multi-dimensional metric used for exposing
// a certain type of container statistic.
type containerMetric struct {
	name        string
	help        string
	valueType   prometheus.ValueType
	extraLabels []string
	getValues   func(s *info.ContainerStats) metricValues
}

func (cm *containerMetric) desc() *prometheus.Desc {
	return prometheus.NewDesc(cm.name, cm.help, append([]string{"name", "id"}, cm.extraLabels...), nil)
}

// PrometheusCollector implements prometheus.Collector.
type PrometheusCollector struct {
	infoProvider     subcontainersInfoProvider
	errors           prometheus.Gauge
	containerMetrics []containerMetric
}

// NewPrometheusCollector returns a new PrometheusCollector.
func NewPrometheusCollector(infoProvider subcontainersInfoProvider) *PrometheusCollector {
	c := &PrometheusCollector{
		infoProvider: infoProvider,
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
						values = append(values, metricValue{
							value:  float64(value) / float64(time.Second),
							labels: []string{fmt.Sprintf("cpu%02d", i)},
						})
					}
					return values
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
				name:        "container_fs_reads_total",
				help:        "Cumulative count of reads completed",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.ReadsCompleted)
					})
				},
			}, {
				name:        "container_fs_sector_reads_total",
				help:        "Cumulative count of sector reads completed",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.SectorsRead)
					})
				},
			}, {
				name:        "container_fs_reads_merged_total",
				help:        "Cumulative count of reads merged",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.ReadsMerged)
					})
				},
			}, {
				name:        "container_fs_read_seconds_total",
				help:        "Cumulative count of seconds spent reading",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.ReadTime) / float64(time.Second)
					})
				},
			}, {
				name:        "container_fs_writes_total",
				help:        "Cumulative count of writes completed",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.WritesCompleted)
					})
				},
			}, {
				name:        "container_fs_sector_writes_total",
				help:        "Cumulative count of sector writes completed",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.SectorsWritten)
					})
				},
			}, {
				name:        "container_fs_writes_merged_total",
				help:        "Cumulative count of writes merged",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.WritesMerged)
					})
				},
			}, {
				name:        "container_fs_write_seconds_total",
				help:        "Cumulative count of seconds spent writing",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.WriteTime) / float64(time.Second)
					})
				},
			}, {
				name:        "container_fs_io_current",
				help:        "Number of I/Os currently in progress",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(fs.IoInProgress)
					})
				},
			}, {
				name:        "container_fs_io_time_seconds_total",
				help:        "Cumulative count of seconds spent doing I/Os",
				valueType:   prometheus.CounterValue,
				extraLabels: []string{"device"},
				getValues: func(s *info.ContainerStats) metricValues {
					return fsValues(s.Filesystem, func(fs *info.FsStats) float64 {
						return float64(float64(fs.IoTime) / float64(time.Second))
					})
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
				name:      "container_network_receive_bytes_total",
				help:      "Cumulative count of bytes received",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Network.RxBytes)}}
				},
			}, {
				name:      "container_network_receive_packets_total",
				help:      "Cumulative count of packets received",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Network.RxPackets)}}
				},
			}, {
				name:      "container_network_receive_packets_dropped_total",
				help:      "Cumulative count of packets dropped while receiving",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Network.RxDropped)}}
				},
			}, {
				name:      "container_network_receive_errors_total",
				help:      "Cumulative count of errors encountered while receiving",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Network.RxErrors)}}
				},
			}, {
				name:      "container_network_transmit_bytes_total",
				help:      "Cumulative count of bytes transmitted",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Network.TxBytes)}}
				},
			}, {
				name:      "container_network_transmit_packets_total",
				help:      "Cumulative count of packets transmitted",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Network.TxPackets)}}
				},
			}, {
				name:      "container_network_transmit_packets_dropped_total",
				help:      "Cumulative count of packets dropped while transmitting",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Network.TxDropped)}}
				},
			}, {
				name:      "container_network_transmit_errors_total",
				help:      "Cumulative count of errors encountered while transmitting",
				valueType: prometheus.CounterValue,
				getValues: func(s *info.ContainerStats) metricValues {
					return metricValues{{value: float64(s.Network.TxErrors)}}
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

// Describe describes all the metrics ever exported by cadvisor. It
// implements prometheus.PrometheusCollector.
func (c *PrometheusCollector) Describe(ch chan<- *prometheus.Desc) {
	c.errors.Describe(ch)
	for _, cm := range c.containerMetrics {
		ch <- cm.desc()
	}
}

// Collect fetches the stats from all containers and delivers them as
// Prometheus metrics. It implements prometheus.PrometheusCollector.
func (c *PrometheusCollector) Collect(ch chan<- prometheus.Metric) {
	containers, err := c.infoProvider.SubcontainersInfo("/", &info.ContainerInfoRequest{NumStats: 1})
	if err != nil {
		c.errors.Set(1)
		glog.Warningf("Couldn't get containers: %s", err)
		return
	}
	for _, container := range containers {
		id := container.Name
		name := id
		if len(container.Aliases) > 0 {
			name = container.Aliases[0]
		}
		stats := container.Stats[0]

		for _, cm := range c.containerMetrics {
			desc := cm.desc()
			for _, metricValue := range cm.getValues(stats) {
				ch <- prometheus.MustNewConstMetric(desc, cm.valueType, float64(metricValue.value), append([]string{name, id}, metricValue.labels...)...)
			}
		}
	}
	c.errors.Collect(ch)
}
