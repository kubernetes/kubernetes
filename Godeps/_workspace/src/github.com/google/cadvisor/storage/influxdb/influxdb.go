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

package influxdb

import (
	"fmt"
	"sync"
	"time"

	info "github.com/google/cadvisor/info/v1"
	influxdb "github.com/influxdb/influxdb/client"
)

type influxdbStorage struct {
	client         *influxdb.Client
	machineName    string
	tableName      string
	bufferDuration time.Duration
	lastWrite      time.Time
	series         []*influxdb.Series
	lock           sync.Mutex
	readyToFlush   func() bool
}

const (
	colTimestamp          string = "time"
	colMachineName        string = "machine"
	colContainerName      string = "container_name"
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
	// Filesystem device.
	colFsDevice = "fs_device"
	// Filesystem limit.
	colFsLimit = "fs_limit"
	// Filesystem usage.
	colFsUsage = "fs_usage"
)

func (self *influxdbStorage) getSeriesDefaultValues(
	ref info.ContainerReference,
	stats *info.ContainerStats,
	columns *[]string,
	values *[]interface{}) {
	// Timestamp
	*columns = append(*columns, colTimestamp)
	*values = append(*values, stats.Timestamp.UnixNano()/1E3)

	// Machine name
	*columns = append(*columns, colMachineName)
	*values = append(*values, self.machineName)

	// Container name
	*columns = append(*columns, colContainerName)
	if len(ref.Aliases) > 0 {
		*values = append(*values, ref.Aliases[0])
	} else {
		*values = append(*values, ref.Name)
	}
}

// In order to maintain a fixed column format, we add a new series for each filesystem partition.
func (self *influxdbStorage) containerFilesystemStatsToSeries(
	ref info.ContainerReference,
	stats *info.ContainerStats) (series []*influxdb.Series) {
	if len(stats.Filesystem) == 0 {
		return series
	}
	for _, fsStat := range stats.Filesystem {
		columns := make([]string, 0)
		values := make([]interface{}, 0)
		self.getSeriesDefaultValues(ref, stats, &columns, &values)

		columns = append(columns, colFsDevice)
		values = append(values, fsStat.Device)

		columns = append(columns, colFsLimit)
		values = append(values, fsStat.Limit)

		columns = append(columns, colFsUsage)
		values = append(values, fsStat.Usage)
		series = append(series, self.newSeries(columns, values))
	}
	return series
}

func (self *influxdbStorage) containerStatsToValues(
	ref info.ContainerReference,
	stats *info.ContainerStats,
) (columns []string, values []interface{}) {
	self.getSeriesDefaultValues(ref, stats, &columns, &values)
	// Cumulative Cpu Usage
	columns = append(columns, colCpuCumulativeUsage)
	values = append(values, stats.Cpu.Usage.Total)

	// Memory Usage
	columns = append(columns, colMemoryUsage)
	values = append(values, stats.Memory.Usage)

	// Working set size
	columns = append(columns, colMemoryWorkingSet)
	values = append(values, stats.Memory.WorkingSet)

	// Network stats.
	columns = append(columns, colRxBytes)
	values = append(values, stats.Network.RxBytes)

	columns = append(columns, colRxErrors)
	values = append(values, stats.Network.RxErrors)

	columns = append(columns, colTxBytes)
	values = append(values, stats.Network.TxBytes)

	columns = append(columns, colTxErrors)
	values = append(values, stats.Network.TxErrors)

	return columns, values
}

func (self *influxdbStorage) OverrideReadyToFlush(readyToFlush func() bool) {
	self.readyToFlush = readyToFlush
}

func (self *influxdbStorage) defaultReadyToFlush() bool {
	return time.Since(self.lastWrite) >= self.bufferDuration
}

func (self *influxdbStorage) AddStats(ref info.ContainerReference, stats *info.ContainerStats) error {
	if stats == nil {
		return nil
	}
	var seriesToFlush []*influxdb.Series
	func() {
		// AddStats will be invoked simultaneously from multiple threads and only one of them will perform a write.
		self.lock.Lock()
		defer self.lock.Unlock()

		self.series = append(self.series, self.newSeries(self.containerStatsToValues(ref, stats)))
		self.series = append(self.series, self.containerFilesystemStatsToSeries(ref, stats)...)
		if self.readyToFlush() {
			seriesToFlush = self.series
			self.series = make([]*influxdb.Series, 0)
			self.lastWrite = time.Now()
		}
	}()
	if len(seriesToFlush) > 0 {
		err := self.client.WriteSeriesWithTimePrecision(seriesToFlush, influxdb.Microsecond)
		if err != nil {
			return fmt.Errorf("failed to write stats to influxDb - %s", err)
		}
	}

	return nil
}

func (self *influxdbStorage) Close() error {
	self.client = nil
	return nil
}

// Returns a new influxdb series.
func (self *influxdbStorage) newSeries(columns []string, points []interface{}) *influxdb.Series {
	out := &influxdb.Series{
		Name:    self.tableName,
		Columns: columns,
		// There's only one point for each stats
		Points: make([][]interface{}, 1),
	}
	out.Points[0] = points
	return out
}

// machineName: A unique identifier to identify the host that current cAdvisor
// instance is running on.
// influxdbHost: The host which runs influxdb.
func New(machineName,
	tablename,
	database,
	username,
	password,
	influxdbHost string,
	isSecure bool,
	bufferDuration time.Duration,
) (*influxdbStorage, error) {
	config := &influxdb.ClientConfig{
		Host:     influxdbHost,
		Username: username,
		Password: password,
		Database: database,
		IsSecure: isSecure,
	}
	client, err := influxdb.NewClient(config)
	if err != nil {
		return nil, err
	}
	// TODO(monnand): With go 1.3, we cannot compress data now.
	client.DisableCompression()

	ret := &influxdbStorage{
		client:         client,
		machineName:    machineName,
		tableName:      tablename,
		bufferDuration: bufferDuration,
		lastWrite:      time.Now(),
		series:         make([]*influxdb.Series, 0),
	}
	ret.readyToFlush = ret.defaultReadyToFlush
	return ret, nil
}
