// Copyright 2015 Google Inc. All Rights Reserved.
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

package statsd

import (
	info "github.com/google/cadvisor/info/v1"
	client "github.com/google/cadvisor/storage/statsd/client"
)

type statsdStorage struct {
	client    *client.Client
	Namespace string
}

const (
	colCpuCumulativeUsage string = "cpu_cumulative_usage"
	// Memory Usage
	colMemoryUsage string = "memory_usage"
	// Working set size
	colMemoryWorkingSet string = "memory_working_set"
	// Cumulative count of bytes received.
	colRxBytes string = "rx_bytes"
	// Cumulative count of receive errors encountered.
	colRxErrors string = "rx_errors"
	// Cumulative count of bytes transmitted.
	colTxBytes string = "tx_bytes"
	// Cumulative count of transmit errors encountered.
	colTxErrors string = "tx_errors"
	// Filesystem summary
	colFsSummary = "fs_summary"
	// Filesystem limit.
	colFsLimit = "fs_limit"
	// Filesystem usage.
	colFsUsage = "fs_usage"
)

func (self *statsdStorage) containerStatsToValues(
	stats *info.ContainerStats,
) (series map[string]uint64) {
	series = make(map[string]uint64)

	// Cumulative Cpu Usage
	series[colCpuCumulativeUsage] = stats.Cpu.Usage.Total

	// Memory Usage
	series[colMemoryUsage] = stats.Memory.Usage

	// Working set size
	series[colMemoryWorkingSet] = stats.Memory.WorkingSet

	// Network stats.
	series[colRxBytes] = stats.Network.RxBytes
	series[colRxErrors] = stats.Network.RxErrors
	series[colTxBytes] = stats.Network.TxBytes
	series[colTxErrors] = stats.Network.TxErrors

	return series
}

func (self *statsdStorage) containerFsStatsToValues(
	series *map[string]uint64,
	stats *info.ContainerStats,
) {
	for _, fsStat := range stats.Filesystem {
		// Summary stats.
		(*series)[colFsSummary+"."+colFsLimit] += fsStat.Limit
		(*series)[colFsSummary+"."+colFsUsage] += fsStat.Usage

		// Per device stats.
		(*series)[fsStat.Device+"."+colFsLimit] = fsStat.Limit
		(*series)[fsStat.Device+"."+colFsUsage] = fsStat.Usage
	}
}

//Push the data into redis
func (self *statsdStorage) AddStats(ref info.ContainerReference, stats *info.ContainerStats) error {
	if stats == nil {
		return nil
	}

	var containerName string
	if len(ref.Aliases) > 0 {
		containerName = ref.Aliases[0]
	} else {
		containerName = ref.Name
	}

	series := self.containerStatsToValues(stats)
	self.containerFsStatsToValues(&series, stats)
	for key, value := range series {
		err := self.client.Send(self.Namespace, containerName, key, value)
		if err != nil {
			return err
		}
	}
	return nil
}

func (self *statsdStorage) Close() error {
	self.client.Close()
	self.client = nil
	return nil
}

func New(namespace, hostPort string) (*statsdStorage, error) {
	statsdClient, err := client.New(hostPort)
	if err != nil {
		return nil, err
	}
	statsdStorage := &statsdStorage{
		client:    statsdClient,
		Namespace: namespace,
	}
	return statsdStorage, nil
}
