// Copyright 2020 Google Inc. All Rights Reserved.
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
	"strconv"

	"github.com/google/cadvisor/container"
	info "github.com/google/cadvisor/info/v1"
	"github.com/prometheus/client_golang/prometheus"

	"k8s.io/klog/v2"
)

var baseLabelsNames = []string{"machine_id", "system_uuid", "boot_id"}

const (
	prometheusModeLabelName     = "mode"
	prometheusTypeLabelName     = "type"
	prometheusLevelLabelName    = "level"
	prometheusNodeLabelName     = "node_id"
	prometheusCoreLabelName     = "core_id"
	prometheusThreadLabelName   = "thread_id"
	prometheusPageSizeLabelName = "page_size"

	nvmMemoryMode    = "memory_mode"
	nvmAppDirectMode = "app_direct_mode"

	memoryByTypeDimmCountKey    = "DimmCount"
	memoryByTypeDimmCapacityKey = "Capacity"

	emptyLabelValue = ""
)

// machineMetric describes a multi-dimensional metric used for exposing a
// certain type of machine statistic.
type machineMetric struct {
	name        string
	help        string
	valueType   prometheus.ValueType
	extraLabels []string
	condition   func(machineInfo *info.MachineInfo) bool
	getValues   func(machineInfo *info.MachineInfo) metricValues
}

func (metric *machineMetric) desc(baseLabels []string) *prometheus.Desc {
	return prometheus.NewDesc(metric.name, metric.help, append(baseLabels, metric.extraLabels...), nil)
}

// PrometheusMachineCollector implements prometheus.Collector.
type PrometheusMachineCollector struct {
	infoProvider   infoProvider
	errors         prometheus.Gauge
	machineMetrics []machineMetric
}

// NewPrometheusMachineCollector returns a new PrometheusCollector.
func NewPrometheusMachineCollector(i infoProvider, includedMetrics container.MetricSet) *PrometheusMachineCollector {
	c := &PrometheusMachineCollector{

		infoProvider: i,
		errors: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: "machine",
			Name:      "scrape_error",
			Help:      "1 if there was an error while getting machine metrics, 0 otherwise.",
		}),
		machineMetrics: []machineMetric{
			{
				name:      "machine_cpu_physical_cores",
				help:      "Number of physical CPU cores.",
				valueType: prometheus.GaugeValue,
				getValues: func(machineInfo *info.MachineInfo) metricValues {
					return metricValues{{value: float64(machineInfo.NumPhysicalCores), timestamp: machineInfo.Timestamp}}
				},
			},
			{
				name:      "machine_cpu_cores",
				help:      "Number of logical CPU cores.",
				valueType: prometheus.GaugeValue,
				getValues: func(machineInfo *info.MachineInfo) metricValues {
					return metricValues{{value: float64(machineInfo.NumCores), timestamp: machineInfo.Timestamp}}
				},
			},
			{
				name:      "machine_cpu_sockets",
				help:      "Number of CPU sockets.",
				valueType: prometheus.GaugeValue,
				getValues: func(machineInfo *info.MachineInfo) metricValues {
					return metricValues{{value: float64(machineInfo.NumSockets), timestamp: machineInfo.Timestamp}}
				},
			},
			{
				name:      "machine_memory_bytes",
				help:      "Amount of memory installed on the machine.",
				valueType: prometheus.GaugeValue,
				getValues: func(machineInfo *info.MachineInfo) metricValues {
					return metricValues{{value: float64(machineInfo.MemoryCapacity), timestamp: machineInfo.Timestamp}}
				},
			},
			{
				name:        "machine_dimm_count",
				help:        "Number of RAM DIMM (all types memory modules) value labeled by dimm type.",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{prometheusTypeLabelName},
				condition:   func(machineInfo *info.MachineInfo) bool { return len(machineInfo.MemoryByType) != 0 },
				getValues: func(machineInfo *info.MachineInfo) metricValues {
					return getMemoryByType(machineInfo, memoryByTypeDimmCountKey)
				},
			},
			{
				name:        "machine_dimm_capacity_bytes",
				help:        "Total RAM DIMM capacity (all types memory modules) value labeled by dimm type.",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{prometheusTypeLabelName},
				condition:   func(machineInfo *info.MachineInfo) bool { return len(machineInfo.MemoryByType) != 0 },
				getValues: func(machineInfo *info.MachineInfo) metricValues {
					return getMemoryByType(machineInfo, memoryByTypeDimmCapacityKey)
				},
			},
			{
				name:        "machine_nvm_capacity",
				help:        "NVM capacity value labeled by NVM mode (memory mode or app direct mode).",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{prometheusModeLabelName},
				getValues: func(machineInfo *info.MachineInfo) metricValues {
					return metricValues{
						{value: float64(machineInfo.NVMInfo.MemoryModeCapacity), labels: []string{nvmMemoryMode}, timestamp: machineInfo.Timestamp},
						{value: float64(machineInfo.NVMInfo.AppDirectModeCapacity), labels: []string{nvmAppDirectMode}, timestamp: machineInfo.Timestamp},
					}
				},
			},
			{
				name:      "machine_nvm_avg_power_budget_watts",
				help:      "NVM power budget.",
				valueType: prometheus.GaugeValue,
				getValues: func(machineInfo *info.MachineInfo) metricValues {
					return metricValues{{value: float64(machineInfo.NVMInfo.AvgPowerBudget), timestamp: machineInfo.Timestamp}}
				},
			},
		},
	}

	if includedMetrics.Has(container.CPUTopologyMetrics) {
		c.machineMetrics = append(c.machineMetrics, []machineMetric{
			{
				name:        "machine_cpu_cache_capacity_bytes",
				help:        "Cache size in bytes assigned to NUMA node and CPU core.",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{prometheusNodeLabelName, prometheusCoreLabelName, prometheusTypeLabelName, prometheusLevelLabelName},
				getValues: func(machineInfo *info.MachineInfo) metricValues {
					return getCaches(machineInfo)
				},
			},
			{
				name:        "machine_thread_siblings_count",
				help:        "Number of CPU thread siblings.",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{prometheusNodeLabelName, prometheusCoreLabelName, prometheusThreadLabelName},
				getValues: func(machineInfo *info.MachineInfo) metricValues {
					return getThreadsSiblingsCount(machineInfo)
				},
			},
			{
				name:        "machine_node_memory_capacity_bytes",
				help:        "Amount of memory assigned to NUMA node.",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{prometheusNodeLabelName},
				getValues: func(machineInfo *info.MachineInfo) metricValues {
					return getNodeMemory(machineInfo)
				},
			},
			{
				name:        "machine_node_hugepages_count",
				help:        "Numer of hugepages assigned to NUMA node.",
				valueType:   prometheus.GaugeValue,
				extraLabels: []string{prometheusNodeLabelName, prometheusPageSizeLabelName},
				getValues: func(machineInfo *info.MachineInfo) metricValues {
					return getHugePagesCount(machineInfo)
				},
			},
		}...)
	}
	return c
}

// Describe describes all the machine metrics ever exported by cadvisor. It
// implements prometheus.PrometheusCollector.
func (collector *PrometheusMachineCollector) Describe(ch chan<- *prometheus.Desc) {
	collector.errors.Describe(ch)
	for _, metric := range collector.machineMetrics {
		ch <- metric.desc([]string{})
	}
}

// Collect fetches information about machine and delivers them as
// Prometheus metrics. It implements prometheus.PrometheusCollector.
func (collector *PrometheusMachineCollector) Collect(ch chan<- prometheus.Metric) {
	collector.errors.Set(0)
	collector.collectMachineInfo(ch)
	collector.errors.Collect(ch)
}

func (collector *PrometheusMachineCollector) collectMachineInfo(ch chan<- prometheus.Metric) {
	machineInfo, err := collector.infoProvider.GetMachineInfo()
	if err != nil {
		collector.errors.Set(1)
		klog.Warningf("Couldn't get machine info: %s", err)
		return
	}

	baseLabelsValues := []string{machineInfo.MachineID, machineInfo.SystemUUID, machineInfo.BootID}

	for _, metric := range collector.machineMetrics {
		if metric.condition != nil && !metric.condition(machineInfo) {
			continue
		}

		for _, metricValue := range metric.getValues(machineInfo) {
			labelValues := make([]string, len(baseLabelsValues))
			copy(labelValues, baseLabelsValues)
			if len(metric.extraLabels) != 0 {
				labelValues = append(labelValues, metricValue.labels...)
			}

			prometheusMetric := prometheus.MustNewConstMetric(metric.desc(baseLabelsNames),
				metric.valueType, metricValue.value, labelValues...)

			if metricValue.timestamp.IsZero() {
				ch <- prometheusMetric
			} else {
				ch <- prometheus.NewMetricWithTimestamp(metricValue.timestamp, prometheusMetric)
			}
		}

	}
}

func getMemoryByType(machineInfo *info.MachineInfo, property string) metricValues {
	mValues := make(metricValues, 0, len(machineInfo.MemoryByType))
	for memoryType, memoryInfo := range machineInfo.MemoryByType {
		propertyValue := 0.0
		switch property {
		case memoryByTypeDimmCapacityKey:
			propertyValue = float64(memoryInfo.Capacity)
		case memoryByTypeDimmCountKey:
			propertyValue = float64(memoryInfo.DimmCount)
		default:
			klog.Warningf("Incorrect propery name for MemoryByType, property %s", property)
			return metricValues{}
		}
		mValues = append(mValues, metricValue{value: propertyValue, labels: []string{memoryType}, timestamp: machineInfo.Timestamp})
	}
	return mValues
}

func getThreadsSiblingsCount(machineInfo *info.MachineInfo) metricValues {
	mValues := make(metricValues, 0, machineInfo.NumCores)
	for _, node := range machineInfo.Topology {
		nodeID := strconv.Itoa(node.Id)

		for _, core := range node.Cores {
			coreID := strconv.Itoa(core.Id)
			siblingsCount := len(core.Threads)

			for _, thread := range core.Threads {
				mValues = append(mValues,
					metricValue{
						value:     float64(siblingsCount),
						labels:    []string{nodeID, coreID, strconv.Itoa(thread)},
						timestamp: machineInfo.Timestamp,
					})
			}
		}
	}
	return mValues
}

func getNodeMemory(machineInfo *info.MachineInfo) metricValues {
	mValues := make(metricValues, 0, len(machineInfo.Topology))
	for _, node := range machineInfo.Topology {
		nodeID := strconv.Itoa(node.Id)
		mValues = append(mValues,
			metricValue{
				value:     float64(node.Memory),
				labels:    []string{nodeID},
				timestamp: machineInfo.Timestamp,
			})
	}
	return mValues
}

func getHugePagesCount(machineInfo *info.MachineInfo) metricValues {
	mValues := make(metricValues, 0)
	for _, node := range machineInfo.Topology {
		nodeID := strconv.Itoa(node.Id)

		for _, hugePage := range node.HugePages {
			mValues = append(mValues,
				metricValue{
					value:     float64(hugePage.NumPages),
					labels:    []string{nodeID, strconv.FormatUint(hugePage.PageSize, 10)},
					timestamp: machineInfo.Timestamp,
				})
		}
	}
	return mValues
}

func getCaches(machineInfo *info.MachineInfo) metricValues {
	mValues := make(metricValues, 0)
	for _, node := range machineInfo.Topology {
		nodeID := strconv.Itoa(node.Id)

		for _, core := range node.Cores {
			coreID := strconv.Itoa(core.Id)

			for _, cache := range core.Caches {
				mValues = append(mValues,
					metricValue{
						value:     float64(cache.Size),
						labels:    []string{nodeID, coreID, cache.Type, strconv.Itoa(cache.Level)},
						timestamp: machineInfo.Timestamp,
					})
			}
		}

		for _, cache := range node.Caches {
			mValues = append(mValues,
				metricValue{
					value:     float64(cache.Size),
					labels:    []string{nodeID, emptyLabelValue, cache.Type, strconv.Itoa(cache.Level)},
					timestamp: machineInfo.Timestamp,
				})
		}
	}
	return mValues
}
