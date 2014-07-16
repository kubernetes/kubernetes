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
	"strconv"
	"strings"
	"time"

	"github.com/google/cadvisor/info"
	"github.com/google/cadvisor/storage"
	"github.com/influxdb/influxdb-go"
)

type influxdbStorage struct {
	client      *influxdb.Client
	prevStats   *info.ContainerStats
	machineName string
	tableName   string
	windowLen   time.Duration
}

const (
	colTimestamp          string = "timestamp"
	colMachineName        string = "machine"
	colContainerName      string = "container_name"
	colCpuCumulativeUsage string = "cpu_cumulative_usage"
	// Cumulative Cpu Usage in system mode
	colCpuCumulativeUsageSystem string = "cpu_cumulative_usage_system"
	// Cumulative Cpu Usage in user mode
	colCpuCumulativeUsageUser string = "cpu_cumulative_usage_user"
	// Memory Usage
	colMemoryUsage string = "memory_usage"
	// Working set size
	colMemoryWorkingSet string = "memory_working_set"
	// container page fault
	colMemoryContainerPgfault string = "memory_container_pgfault"
	// container major page fault
	colMemoryContainerPgmajfault string = "memory_container_pgmajfault"
	// hierarchical page fault
	colMemoryHierarchicalPgfault string = "memory_hierarchical_pgfault"
	// hierarchical major page fault
	colMemoryHierarchicalPgmajfault string = "memory_hierarchical_pgmajfault"
	// Cumulative per core usage
	colPerCoreCumulativeUsagePrefix string = "per_core_cumulative_usage_core_"
	// Optional: sample duration. Unit: Nanosecond.
	colSampleDuration string = "sample_duration"
	// Optional: Instant cpu usage
	colCpuInstantUsage string = "cpu_instant_usage"
	// Optional: Instant per core usage
	colPerCoreInstantUsagePrefix string = "per_core_instant_usage_core_"
)

func (self *influxdbStorage) containerStatsToValues(
	ref info.ContainerReference,
	stats *info.ContainerStats,
) (columns []string, values []interface{}) {

	// Timestamp
	columns = append(columns, colTimestamp)
	values = append(values, stats.Timestamp.Format(time.RFC3339Nano))

	// Machine name
	columns = append(columns, colMachineName)
	values = append(values, self.machineName)

	// Container name
	columns = append(columns, colContainerName)
	values = append(values, ref.Name)

	// Cumulative Cpu Usage
	columns = append(columns, colCpuCumulativeUsage)
	values = append(values, stats.Cpu.Usage.Total)

	// Cumulative Cpu Usage in system mode
	columns = append(columns, colCpuCumulativeUsageSystem)
	values = append(values, stats.Cpu.Usage.System)

	// Cumulative Cpu Usage in user mode
	columns = append(columns, colCpuCumulativeUsageUser)
	values = append(values, stats.Cpu.Usage.User)

	// Memory Usage
	columns = append(columns, colMemoryUsage)
	values = append(values, stats.Memory.Usage)

	// Working set size
	columns = append(columns, colMemoryWorkingSet)
	values = append(values, stats.Memory.WorkingSet)

	// container page fault
	columns = append(columns, colMemoryContainerPgfault)
	values = append(values, stats.Memory.ContainerData.Pgfault)

	// container major page fault
	columns = append(columns, colMemoryContainerPgmajfault)
	values = append(values, stats.Memory.ContainerData.Pgmajfault)

	// hierarchical page fault
	columns = append(columns, colMemoryHierarchicalPgfault)
	values = append(values, stats.Memory.HierarchicalData.Pgfault)

	// hierarchical major page fault
	columns = append(columns, colMemoryHierarchicalPgmajfault)
	values = append(values, stats.Memory.HierarchicalData.Pgmajfault)

	// per cpu cumulative usage
	for i, u := range stats.Cpu.Usage.PerCpu {
		columns = append(columns, fmt.Sprintf("%v%v", colPerCoreCumulativeUsagePrefix, i))
		values = append(values, u)
	}

	sample, err := info.NewSample(self.prevStats, stats)
	if err != nil || sample == nil {
		return columns, values
	}

	// Optional: sample duration. Unit: Nanosecond.
	columns = append(columns, colSampleDuration)
	values = append(values, sample.Duration.String())

	// Optional: Instant cpu usage
	columns = append(columns, colCpuInstantUsage)
	values = append(values, sample.Cpu.Usage)

	// Optional: Instant per core usage
	for i, u := range sample.Cpu.PerCpuUsage {
		columns = append(columns, fmt.Sprintf("%v%v", colPerCoreInstantUsagePrefix, i))
		values = append(values, u)
	}

	return columns, values
}

func convertToUint64(v interface{}) (uint64, error) {
	if v == nil {
		return 0, nil
	}
	switch x := v.(type) {
	case uint64:
		return x, nil
	case int:
		if x < 0 {
			return 0, fmt.Errorf("negative value: %v", x)
		}
		return uint64(x), nil
	case int32:
		if x < 0 {
			return 0, fmt.Errorf("negative value: %v", x)
		}
		return uint64(x), nil
	case int64:
		if x < 0 {
			return 0, fmt.Errorf("negative value: %v", x)
		}
		return uint64(x), nil
	case float64:
		if x < 0 {
			return 0, fmt.Errorf("negative value: %v", x)
		}
		return uint64(x), nil
	case uint32:
		return uint64(x), nil
	}
	return 0, fmt.Errorf("Unknown type")
}

func (self *influxdbStorage) valuesToContainerStats(columns []string, values []interface{}) (*info.ContainerStats, error) {
	stats := &info.ContainerStats{
		Cpu:    &info.CpuStats{},
		Memory: &info.MemoryStats{},
	}
	perCoreUsage := make(map[int]uint64, 32)
	var err error
	for i, col := range columns {
		v := values[i]
		switch {
		case col == colTimestamp:
			if str, ok := v.(string); ok {
				stats.Timestamp, err = time.Parse(time.RFC3339Nano, str)
			}
		case col == colMachineName:
			if m, ok := v.(string); ok {
				if m != self.machineName {
					return nil, fmt.Errorf("different machine")
				}
			} else {
				return nil, fmt.Errorf("machine name field is not a string: %v", v)
			}
		// Cumulative Cpu Usage
		case col == colCpuCumulativeUsage:
			stats.Cpu.Usage.Total, err = convertToUint64(v)
		// Cumulative Cpu used by the system
		case col == colCpuCumulativeUsageSystem:
			stats.Cpu.Usage.System, err = convertToUint64(v)
		// Cumulative Cpu Usage in user mode
		case col == colCpuCumulativeUsageUser:
			stats.Cpu.Usage.User, err = convertToUint64(v)
		// Memory Usage
		case col == colMemoryUsage:
			stats.Memory.Usage, err = convertToUint64(v)
		// Working set size
		case col == colMemoryWorkingSet:
			stats.Memory.WorkingSet, err = convertToUint64(v)
		// container page fault
		case col == colMemoryContainerPgfault:
			stats.Memory.ContainerData.Pgfault, err = convertToUint64(v)
		// container major page fault
		case col == colMemoryContainerPgmajfault:
			stats.Memory.ContainerData.Pgmajfault, err = convertToUint64(v)
		// hierarchical page fault
		case col == colMemoryHierarchicalPgfault:
			stats.Memory.HierarchicalData.Pgfault, err = convertToUint64(v)
		// hierarchical major page fault
		case col == colMemoryHierarchicalPgmajfault:
			stats.Memory.HierarchicalData.Pgmajfault, err = convertToUint64(v)
		case strings.HasPrefix(col, colPerCoreCumulativeUsagePrefix):
			idxStr := col[len(colPerCoreCumulativeUsagePrefix):]
			idx, err := strconv.Atoi(idxStr)
			if err != nil {
				continue
			}
			perCoreUsage[idx], err = convertToUint64(v)
		}
		if err != nil {
			return nil, fmt.Errorf("column %v has invalid value %v: %v", col, v, err)
		}
	}
	stats.Cpu.Usage.PerCpu = make([]uint64, len(perCoreUsage))
	for idx, usage := range perCoreUsage {
		stats.Cpu.Usage.PerCpu[idx] = usage
	}
	return stats, nil
}

func (self *influxdbStorage) valuesToContainerSample(columns []string, values []interface{}) (*info.ContainerStatsSample, error) {
	sample := &info.ContainerStatsSample{}
	perCoreUsage := make(map[int]uint64, 32)
	var err error
	for i, col := range columns {
		v := values[i]
		switch {
		case col == colTimestamp:
			if str, ok := v.(string); ok {
				sample.Timestamp, err = time.Parse(time.RFC3339Nano, str)
			}
		case col == colMachineName:
			if m, ok := v.(string); ok {
				if m != self.machineName {
					return nil, fmt.Errorf("different machine")
				}
			} else {
				return nil, fmt.Errorf("machine name field is not a string: %v", v)
			}
		// Memory Usage
		case col == colMemoryUsage:
			sample.Memory.Usage, err = convertToUint64(v)
		// sample duration. Unit: Nanosecond.
		case col == colSampleDuration:
			if v == nil {
				// this record does not have sample_duration, so it's the first stats.
				return nil, nil
			}
			sample.Duration, err = time.ParseDuration(v.(string))
		// Instant cpu usage
		case col == colCpuInstantUsage:
			sample.Cpu.Usage, err = convertToUint64(v)
		case strings.HasPrefix(col, colPerCoreInstantUsagePrefix):
			idxStr := col[len(colPerCoreInstantUsagePrefix):]
			idx, err := strconv.Atoi(idxStr)
			if err != nil {
				continue
			}
			perCoreUsage[idx], err = convertToUint64(v)
		}
		if err != nil {
			return nil, fmt.Errorf("column %v has invalid value %v: %v", col, v, err)
		}
	}
	sample.Cpu.PerCpuUsage = make([]uint64, len(perCoreUsage))
	for idx, usage := range perCoreUsage {
		sample.Cpu.PerCpuUsage[idx] = usage
	}
	if sample.Duration.Nanoseconds() == 0 {
		return nil, nil
	}
	return sample, nil
}

func (self *influxdbStorage) AddStats(ref info.ContainerReference, stats *info.ContainerStats) error {
	series := &influxdb.Series{
		Name: self.tableName,
		// There's only one point for each stats
		Points: make([][]interface{}, 1),
	}
	if stats == nil || stats.Cpu == nil || stats.Memory == nil {
		return nil
	}
	series.Columns, series.Points[0] = self.containerStatsToValues(ref, stats)

	self.prevStats = stats.Copy(self.prevStats)
	err := self.client.WriteSeries([]*influxdb.Series{series})
	if err != nil {
		return err
	}
	return nil
}

func (self *influxdbStorage) RecentStats(containerName string, numStats int) ([]*info.ContainerStats, error) {
	// TODO(dengnan): select only columns that we need
	// TODO(dengnan): escape names
	query := fmt.Sprintf("select * from %v where %v='%v' and %v='%v'", self.tableName, colContainerName, containerName, colMachineName, self.machineName)
	if numStats > 0 {
		query = fmt.Sprintf("%v limit %v", query, numStats)
	}
	series, err := self.client.Query(query)
	if err != nil {
		return nil, err
	}
	statsList := make([]*info.ContainerStats, 0, len(series))
	// By default, influxDB returns data in time descending order.
	// RecentStats() requires stats in time increasing order,
	// so we need to go through from the last one to the first one.
	for i := len(series) - 1; i >= 0; i-- {
		s := series[i]
		for j := len(s.Points) - 1; j >= 0; j-- {
			values := s.Points[j]
			stats, err := self.valuesToContainerStats(s.Columns, values)
			if err != nil {
				return nil, err
			}
			if stats == nil {
				continue
			}
			statsList = append(statsList, stats)
		}
	}
	return statsList, nil
}

func (self *influxdbStorage) Samples(containerName string, numSamples int) ([]*info.ContainerStatsSample, error) {
	// TODO(dengnan): select only columns that we need
	// TODO(dengnan): escape names
	query := fmt.Sprintf("select * from %v where %v='%v' and %v='%v'", self.tableName, colContainerName, containerName, colMachineName, self.machineName)
	if numSamples > 0 {
		query = fmt.Sprintf("%v limit %v", query, numSamples)
	}
	series, err := self.client.Query(query)
	if err != nil {
		return nil, err
	}
	sampleList := make([]*info.ContainerStatsSample, 0, len(series))
	for i := len(series) - 1; i >= 0; i-- {
		s := series[i]
		for j := len(s.Points) - 1; j >= 0; j-- {
			values := s.Points[j]
			sample, err := self.valuesToContainerSample(s.Columns, values)
			if err != nil {
				return nil, err
			}
			if sample == nil {
				continue
			}
			sampleList = append(sampleList, sample)
		}
	}
	return sampleList, nil
}

func (self *influxdbStorage) Close() error {
	self.client = nil
	return nil
}

func (self *influxdbStorage) Percentiles(
	containerName string,
	cpuUsagePercentiles []int,
	memUsagePercentiles []int,
) (*info.ContainerStatsPercentiles, error) {
	selectedCol := make([]string, 0, len(cpuUsagePercentiles)+len(memUsagePercentiles)+1)

	selectedCol = append(selectedCol, fmt.Sprintf("max(%v)", colMemoryUsage))
	for _, p := range cpuUsagePercentiles {
		selectedCol = append(selectedCol, fmt.Sprintf("percentile(%v, %v)", colCpuInstantUsage, p))
	}
	for _, p := range memUsagePercentiles {
		selectedCol = append(selectedCol, fmt.Sprintf("percentile(%v, %v)", colMemoryUsage, p))
	}

	query := fmt.Sprintf("select %v from %v where %v='%v' and %v='%v' and time > now() - %v",
		strings.Join(selectedCol, ","),
		self.tableName,
		colContainerName,
		containerName,
		colMachineName,
		self.machineName,
		fmt.Sprintf("%vs", self.windowLen.Seconds()),
	)
	series, err := self.client.Query(query)
	if err != nil {
		return nil, err
	}
	if len(series) != 1 {
		return nil, nil
	}
	if len(series[0].Points) == 0 {
		return nil, nil
	}

	point := series[0].Points[0]

	ret := new(info.ContainerStatsPercentiles)
	ret.MaxMemoryUsage, err = convertToUint64(point[1])
	if err != nil {
		return nil, fmt.Errorf("invalid max memory usage: %v", err)
	}
	retrievedCpuPercentiles := point[2 : 2+len(cpuUsagePercentiles)]
	for i, p := range cpuUsagePercentiles {
		v, err := convertToUint64(retrievedCpuPercentiles[i])
		if err != nil {
			return nil, fmt.Errorf("invalid cpu usage: %v", err)
		}
		ret.CpuUsagePercentiles = append(
			ret.CpuUsagePercentiles,
			info.Percentile{
				Percentage: p,
				Value:      v,
			},
		)
	}
	retrievedMemoryPercentiles := point[2+len(cpuUsagePercentiles):]
	for i, p := range memUsagePercentiles {
		v, err := convertToUint64(retrievedMemoryPercentiles[i])
		if err != nil {
			return nil, fmt.Errorf("invalid memory usage: %v", err)
		}
		ret.MemoryUsagePercentiles = append(
			ret.MemoryUsagePercentiles,
			info.Percentile{
				Percentage: p,
				Value:      v,
			},
		)
	}
	return ret, nil
}

// machineName: A unique identifier to identify the host that current cAdvisor
// instance is running on.
// influxdbHost: The host which runs influxdb.
// percentilesDuration: Time window which will be considered when calls Percentiles()
func New(machineName,
	tablename,
	database,
	username,
	password,
	influxdbHost string,
	isSecure bool,
	percentilesDuration time.Duration,
) (storage.StorageDriver, error) {
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
	if percentilesDuration.Seconds() < 1.0 {
		percentilesDuration = 5 * time.Minute
	}

	ret := &influxdbStorage{
		client:      client,
		windowLen:   percentilesDuration,
		machineName: machineName,
		tableName:   tablename,
	}
	return ret, nil
}
