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
	"github.com/google/cadvisor/manager"
	"github.com/prometheus/client_golang/prometheus"
)

type prometheusMetric struct {
	valueType prometheus.ValueType
	value     float64
	labels    []string
}

// PrometheusCollector implements prometheus.Collector.
type PrometheusCollector struct {
	manager manager.Manager

	errors   prometheus.Gauge
	lastSeen *prometheus.Desc

	cpuUsageUserSeconds   *prometheus.Desc
	cpuUsageSystemSeconds *prometheus.Desc
	cpuUsageSecondsPerCPU *prometheus.Desc

	memoryUsageBytes *prometheus.Desc
	memoryWorkingSet *prometheus.Desc
	memoryFailures   *prometheus.Desc

	fsLimit        *prometheus.Desc
	fsUsage        *prometheus.Desc
	fsReads        *prometheus.Desc
	fsReadsSectors *prometheus.Desc
	fsReadsMerged  *prometheus.Desc
	fsReadTime     *prometheus.Desc

	fsWrites        *prometheus.Desc
	fsWritesSectors *prometheus.Desc
	fsWritesMerged  *prometheus.Desc
	fsWriteTime     *prometheus.Desc

	fsIoInProgress *prometheus.Desc
	fsIoTime       *prometheus.Desc

	fsWeightedIoTime *prometheus.Desc

	networkRxBytes   *prometheus.Desc
	networkRxPackets *prometheus.Desc
	networkRxErrors  *prometheus.Desc
	networkRxDropped *prometheus.Desc
	networkTxBytes   *prometheus.Desc
	networkTxPackets *prometheus.Desc
	networkTxErrors  *prometheus.Desc
	networkTxDropped *prometheus.Desc

	tasks *prometheus.Desc

	descs []*prometheus.Desc
}

// NewPrometheusCollector returns a new PrometheusCollector.
func NewPrometheusCollector(manager manager.Manager) *PrometheusCollector {
	c := &PrometheusCollector{
		manager: manager,
		errors: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: "container",
			Name:      "scrape_error",
			Help:      "1 if there was an error while getting container metrics, 0 otherwise",
		}),
		lastSeen: prometheus.NewDesc(
			"container_last_seen",
			"Last time a container was seen by the exporter",
			[]string{"name", "id"},
			nil),
		cpuUsageUserSeconds: prometheus.NewDesc(
			"container_cpu_user_seconds_total",
			"Cumulative user cpu time consumed in seconds.",
			[]string{"name", "id"},
			nil),
		cpuUsageSystemSeconds: prometheus.NewDesc(
			"container_cpu_system_seconds_total",
			"Cumulative system cpu time consumed in seconds.",
			[]string{"name", "id"},
			nil),
		cpuUsageSecondsPerCPU: prometheus.NewDesc(
			"container_cpu_usage_seconds_total",
			"Cumulative cpu time consumed per cpu in seconds.",
			[]string{"name", "id", "cpu"},
			nil),
		memoryUsageBytes: prometheus.NewDesc(
			"container_memory_usage_bytes",
			"Current memory usage in bytes.",
			[]string{"name", "id"},
			nil),
		memoryWorkingSet: prometheus.NewDesc(
			"container_memory_working_set_bytes",
			"Current working set in bytes.",
			[]string{"name", "id"},
			nil),
		memoryFailures: prometheus.NewDesc(
			"container_memory_failures_total",
			"Cumulative count of memory allocation failures.",
			[]string{"type", "scope", "name", "id"},
			nil),

		fsLimit: prometheus.NewDesc(
			"container_fs_limit_bytes",
			"Number of bytes that can be consumed by the container on this filesystem.",
			[]string{"name", "id", "device"},
			nil),
		fsUsage: prometheus.NewDesc(
			"container_fs_usage_bytes",
			"Number of bytes that are consumed by the container on this filesystem.",
			[]string{"name", "id", "device"},
			nil),
		fsReads: prometheus.NewDesc(
			"container_fs_reads_total",
			"Cumulative count of reads completed",
			[]string{"name", "id", "device"},
			nil),
		fsReadsSectors: prometheus.NewDesc(
			"container_fs_sector_reads_total",
			"Cumulative count of sector reads completed",
			[]string{"name", "id", "device"},
			nil),
		fsReadsMerged: prometheus.NewDesc(
			"container_fs_reads_merged_total",
			"Cumulative count of reads merged",
			[]string{"name", "id", "device"},
			nil),
		fsReadTime: prometheus.NewDesc(
			"container_fs_read_seconds_total",
			"Cumulative count of seconds spent reading",
			[]string{"name", "id", "device"},
			nil),
		fsWrites: prometheus.NewDesc(
			"container_fs_writes_total",
			"Cumulative count of writes completed",
			[]string{"name", "id", "device"},
			nil),
		fsWritesSectors: prometheus.NewDesc(
			"container_fs_sector_writes_total",
			"Cumulative count of sector writes completed",
			[]string{"name", "id", "device"},
			nil),
		fsWritesMerged: prometheus.NewDesc(
			"container_fs_writes_merged_total",
			"Cumulative count of writes merged",
			[]string{"name", "id", "device"},
			nil),
		fsWriteTime: prometheus.NewDesc(
			"container_fs_write_seconds_total",
			"Cumulative count of seconds spent writing",
			[]string{"name", "id", "device"},
			nil),
		fsIoInProgress: prometheus.NewDesc(
			"container_fs_io_current",
			"Number of I/Os currently in progress",
			[]string{"name", "id", "device"},
			nil),
		fsIoTime: prometheus.NewDesc(
			"container_fs_io_time_seconds_total",
			"Cumulative count of seconds spent doing I/Os",
			[]string{"name", "id", "device"},
			nil),
		fsWeightedIoTime: prometheus.NewDesc(
			"container_fs_io_time_weighted_seconds_total",
			"Cumulative weighted I/O time in seconds",
			[]string{"name", "id", "device"},
			nil),
		networkRxBytes: prometheus.NewDesc(
			"container_network_receive_bytes_total",
			"Cumulative count of bytes received",
			[]string{"name", "id"},
			nil),
		networkRxPackets: prometheus.NewDesc(
			"container_network_receive_packets_total",
			"Cumulative count of packets received",
			[]string{"name", "id"},
			nil),
		networkRxDropped: prometheus.NewDesc(
			"container_network_receive_packets_dropped_total",
			"Cumulative count of packets dropped while receiving",
			[]string{"name", "id"},
			nil),
		networkRxErrors: prometheus.NewDesc(
			"container_network_receive_errors_total",
			"Cumulative count of errors encountered while receiving",
			[]string{"name", "id"},
			nil),
		networkTxBytes: prometheus.NewDesc(
			"container_network_transmit_bytes_total",
			"Cumulative count of bytes transmitted",
			[]string{"name", "id"},
			nil),
		networkTxPackets: prometheus.NewDesc(
			"container_network_transmit_packets_total",
			"Cumulative count of packets transmitted",
			[]string{"name", "id"},
			nil),
		networkTxDropped: prometheus.NewDesc(
			"container_network_transmit_packets_dropped_total",
			"Cumulative count of packets dropped while transmitting",
			[]string{"name", "id"},
			nil),
		networkTxErrors: prometheus.NewDesc(
			"container_network_transmit_errors_total",
			"Cumulative count of errors encountered while transmitting",
			[]string{"name", "id"},
			nil),

		tasks: prometheus.NewDesc(
			"container_tasks_state",
			"Number of tasks in given state",
			[]string{"state", "name", "id"},
			nil),
	}
	c.descs = []*prometheus.Desc{
		c.lastSeen,

		c.cpuUsageUserSeconds,
		c.cpuUsageSystemSeconds,

		c.memoryUsageBytes,
		c.memoryWorkingSet,
		c.memoryFailures,

		c.fsLimit,
		c.fsUsage,
		c.fsReads,
		c.fsReadsSectors,
		c.fsReadsMerged,
		c.fsReadTime,
		c.fsWrites,
		c.fsWritesSectors,
		c.fsWritesMerged,
		c.fsWriteTime,
		c.fsIoInProgress,
		c.fsIoTime,
		c.fsWeightedIoTime,

		c.networkRxBytes,
		c.networkRxPackets,
		c.networkRxErrors,
		c.networkRxDropped,
		c.networkTxBytes,
		c.networkTxPackets,
		c.networkTxErrors,
		c.networkTxDropped,

		c.tasks,
	}
	return c
}

// Describe describes all the metrics ever exported by cadvisor. It
// implements prometheus.PrometheusCollector.
func (c *PrometheusCollector) Describe(ch chan<- *prometheus.Desc) {
	c.errors.Describe(ch)
	for _, d := range c.descs {
		ch <- d
	}
}

// Collect fetches the stats from all containers and delivers them as
// Prometheus metrics. It implements prometheus.PrometheusCollector.
func (c *PrometheusCollector) Collect(ch chan<- prometheus.Metric) {
	containers, err := c.manager.SubcontainersInfo("/", &info.ContainerInfoRequest{NumStats: 1})
	if err != nil {
		c.errors.Set(1)
		glog.Warning("Couldn't get containers: %s", err)
		return
	}
	for _, container := range containers {
		id := container.Name
		name := id
		if len(container.Aliases) > 0 {
			name = container.Aliases[0]
		}
		stats := container.Stats[0]

		for desc, metrics := range map[*prometheus.Desc][]prometheusMetric{
			c.cpuUsageUserSeconds:   {{valueType: prometheus.CounterValue, value: float64(stats.Cpu.Usage.User) / float64(time.Second)}},
			c.cpuUsageSystemSeconds: {{valueType: prometheus.CounterValue, value: float64(stats.Cpu.Usage.System) / float64(time.Second)}},

			c.memoryFailures: {
				{valueType: prometheus.CounterValue, labels: []string{"pgfault", "container"}, value: float64(stats.Memory.ContainerData.Pgfault)},
				{valueType: prometheus.CounterValue, labels: []string{"pgmajfault", "container"}, value: float64(stats.Memory.ContainerData.Pgmajfault)},
				{valueType: prometheus.CounterValue, labels: []string{"pgfault", "hierarchy"}, value: float64(stats.Memory.HierarchicalData.Pgfault)},
				{valueType: prometheus.CounterValue, labels: []string{"pgmajfault", "hierarchy"}, value: float64(stats.Memory.HierarchicalData.Pgmajfault)},
			},
			c.tasks: {
				{valueType: prometheus.GaugeValue, labels: []string{"sleeping"}, value: float64(stats.TaskStats.NrSleeping)},
				{valueType: prometheus.GaugeValue, labels: []string{"running"}, value: float64(stats.TaskStats.NrRunning)},
				{valueType: prometheus.GaugeValue, labels: []string{"stopped"}, value: float64(stats.TaskStats.NrStopped)},
				{valueType: prometheus.GaugeValue, labels: []string{"uninterruptible"}, value: float64(stats.TaskStats.NrUninterruptible)},
				{valueType: prometheus.GaugeValue, labels: []string{"iowaiting"}, value: float64(stats.TaskStats.NrIoWait)},
			},

			c.lastSeen: {{valueType: prometheus.GaugeValue, value: float64(time.Now().Unix())}},

			c.memoryUsageBytes: {{valueType: prometheus.GaugeValue, value: float64(stats.Memory.Usage)}},
			c.memoryWorkingSet: {{valueType: prometheus.GaugeValue, value: float64(stats.Memory.WorkingSet)}},

			c.networkRxBytes:   {{valueType: prometheus.CounterValue, value: float64(stats.Network.RxBytes)}},
			c.networkRxPackets: {{valueType: prometheus.CounterValue, value: float64(stats.Network.RxPackets)}},
			c.networkRxErrors:  {{valueType: prometheus.CounterValue, value: float64(stats.Network.RxErrors)}},
			c.networkRxDropped: {{valueType: prometheus.CounterValue, value: float64(stats.Network.RxDropped)}},
			c.networkTxBytes:   {{valueType: prometheus.CounterValue, value: float64(stats.Network.TxBytes)}},
			c.networkTxPackets: {{valueType: prometheus.CounterValue, value: float64(stats.Network.TxPackets)}},
			c.networkTxErrors:  {{valueType: prometheus.CounterValue, value: float64(stats.Network.TxErrors)}},
			c.networkTxDropped: {{valueType: prometheus.CounterValue, value: float64(stats.Network.TxDropped)}},
		} {
			for _, m := range metrics {
				ch <- prometheus.MustNewConstMetric(desc, prometheus.CounterValue, float64(m.value), append(m.labels, name, id)...)
			}
		}

		// Metrics with dynamic labels
		for i, value := range stats.Cpu.Usage.PerCpu {
			ch <- prometheus.MustNewConstMetric(c.cpuUsageSecondsPerCPU, prometheus.CounterValue, float64(value)/float64(time.Second), name, id, fmt.Sprintf("cpu%02d", i))
		}

		for _, stat := range stats.Filesystem {
			for desc, m := range map[*prometheus.Desc]prometheusMetric{
				c.fsReads:        {valueType: prometheus.CounterValue, value: float64(stat.ReadsCompleted)},
				c.fsReadsSectors: {valueType: prometheus.CounterValue, value: float64(stat.SectorsRead)},
				c.fsReadsMerged:  {valueType: prometheus.CounterValue, value: float64(stat.ReadsMerged)},
				c.fsReadTime:     {valueType: prometheus.CounterValue, value: float64(stat.ReadTime) / float64(time.Second)},

				c.fsWrites:        {valueType: prometheus.CounterValue, value: float64(stat.WritesCompleted)},
				c.fsWritesSectors: {valueType: prometheus.CounterValue, value: float64(stat.SectorsWritten)},
				c.fsWritesMerged:  {valueType: prometheus.CounterValue, value: float64(stat.WritesMerged)},
				c.fsWriteTime:     {valueType: prometheus.CounterValue, value: float64(stat.WriteTime) / float64(time.Second)},

				c.fsIoTime:         {valueType: prometheus.CounterValue, value: float64(stat.IoTime) / float64(time.Second)},
				c.fsWeightedIoTime: {valueType: prometheus.CounterValue, value: float64(stat.WeightedIoTime) / float64(time.Second)},

				c.fsIoInProgress: {valueType: prometheus.GaugeValue, value: float64(stat.IoInProgress)},
				c.fsLimit:        {valueType: prometheus.GaugeValue, value: float64(stat.Limit)},
				c.fsUsage:        {valueType: prometheus.GaugeValue, value: float64(stat.Usage)},
			} {
				ch <- prometheus.MustNewConstMetric(desc, m.valueType, m.value, name, id, stat.Device)
			}
		}
	}
	c.errors.Collect(ch)
}
