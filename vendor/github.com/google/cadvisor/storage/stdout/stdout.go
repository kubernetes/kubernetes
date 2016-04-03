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

package stdout

import (
	"bytes"
	"fmt"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/storage"
)

func init() {
	storage.RegisterStorageDriver("stdout", new)
}

type stdoutStorage struct {
	Namespace string
}

const (
	colCpuCumulativeUsage = "cpu_cumulative_usage"
	// Memory Usage
	colMemoryUsage = "memory_usage"
	// Working set size
	colMemoryWorkingSet = "memory_working_set"
	// Cumulative count of bytes received.
	colRxBytes = "rx_bytes"
	// Cumulative count of receive errors encountered.
	colRxErrors = "rx_errors"
	// Cumulative count of bytes transmitted.
	colTxBytes = "tx_bytes"
	// Cumulative count of transmit errors encountered.
	colTxErrors = "tx_errors"
	// Filesystem summary
	colFsSummary = "fs_summary"
	// Filesystem limit.
	colFsLimit = "fs_limit"
	// Filesystem usage.
	colFsUsage = "fs_usage"
)

func new() (storage.StorageDriver, error) {
	return newStorage(*storage.ArgDbHost)
}

func (driver *stdoutStorage) containerStatsToValues(stats *info.ContainerStats) (series map[string]uint64) {
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

func (driver *stdoutStorage) containerFsStatsToValues(series *map[string]uint64, stats *info.ContainerStats) {
	for _, fsStat := range stats.Filesystem {
		// Summary stats.
		(*series)[colFsSummary+"."+colFsLimit] += fsStat.Limit
		(*series)[colFsSummary+"."+colFsUsage] += fsStat.Usage

		// Per device stats.
		(*series)[fsStat.Device+"."+colFsLimit] = fsStat.Limit
		(*series)[fsStat.Device+"."+colFsUsage] = fsStat.Usage
	}
}

func (driver *stdoutStorage) AddStats(ref info.ContainerReference, stats *info.ContainerStats) error {
	if stats == nil {
		return nil
	}

	containerName := ref.Name
	if len(ref.Aliases) > 0 {
		containerName = ref.Aliases[0]
	}

	var buffer bytes.Buffer
	buffer.WriteString(fmt.Sprintf("cName=%s host=%s", containerName, driver.Namespace))

	series := driver.containerStatsToValues(stats)
	driver.containerFsStatsToValues(&series, stats)
	for key, value := range series {
		buffer.WriteString(fmt.Sprintf(" %s=%v", key, value))
	}

	_, err := fmt.Println(buffer.String())

	return err
}

func (driver *stdoutStorage) Close() error {
	return nil
}

func newStorage(namespace string) (*stdoutStorage, error) {
	stdoutStorage := &stdoutStorage{
		Namespace: namespace,
	}
	return stdoutStorage, nil
}
