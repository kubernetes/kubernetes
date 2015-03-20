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

package bigquery

import (
	"fmt"
	"strconv"
	"time"

	bigquery "code.google.com/p/google-api-go-client/bigquery/v2"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/storage"
	"github.com/google/cadvisor/storage/bigquery/client"
)

type bigqueryStorage struct {
	client      *client.Client
	machineName string
}

const (
	// Bigquery schema types
	typeTimestamp string = "TIMESTAMP"
	typeString    string = "STRING"
	typeInteger   string = "INTEGER"

	colTimestamp          string = "timestamp"
	colMachineName        string = "machine"
	colContainerName      string = "container_name"
	colCpuCumulativeUsage string = "cpu_cumulative_usage"
	// Cumulative Cpu usage in system and user mode
	colCpuCumulativeUsageSystem string = "cpu_cumulative_usage_system"
	colCpuCumulativeUsageUser   string = "cpu_cumulative_usage_user"
	// Memory usage
	colMemoryUsage string = "memory_usage"
	// Working set size
	colMemoryWorkingSet string = "memory_working_set"
	// Container page fault
	colMemoryContainerPgfault string = "memory_container_pgfault"
	// Constainer major page fault
	colMemoryContainerPgmajfault string = "memory_container_pgmajfault"
	// Hierarchical page fault
	colMemoryHierarchicalPgfault string = "memory_hierarchical_pgfault"
	// Hierarchical major page fault
	colMemoryHierarchicalPgmajfault string = "memory_hierarchical_pgmajfault"
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
	// Filesystem available space.
	colFsUsage = "fs_usage"
)

// TODO(jnagal): Infer schema through reflection. (See bigquery/client/example)
func (self *bigqueryStorage) GetSchema() *bigquery.TableSchema {
	fields := make([]*bigquery.TableFieldSchema, 18)
	i := 0
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeTimestamp,
		Name: colTimestamp,
		Mode: "REQUIRED",
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeString,
		Name: colMachineName,
		Mode: "REQUIRED",
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeString,
		Name: colContainerName,
		Mode: "REQUIRED",
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeInteger,
		Name: colCpuCumulativeUsage,
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeInteger,
		Name: colCpuCumulativeUsageSystem,
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeInteger,
		Name: colCpuCumulativeUsageUser,
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeInteger,
		Name: colMemoryUsage,
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeInteger,
		Name: colMemoryWorkingSet,
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeInteger,
		Name: colMemoryContainerPgfault,
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeInteger,
		Name: colMemoryContainerPgmajfault,
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeInteger,
		Name: colMemoryHierarchicalPgfault,
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeInteger,
		Name: colMemoryHierarchicalPgmajfault,
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeInteger,
		Name: colRxBytes,
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeInteger,
		Name: colRxErrors,
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeInteger,
		Name: colTxBytes,
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeInteger,
		Name: colTxErrors,
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeString,
		Name: colFsDevice,
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeInteger,
		Name: colFsLimit,
	}
	i++
	fields[i] = &bigquery.TableFieldSchema{
		Type: typeInteger,
		Name: colFsUsage,
	}
	return &bigquery.TableSchema{
		Fields: fields,
	}
}

func (self *bigqueryStorage) containerStatsToRows(
	ref info.ContainerReference,
	stats *info.ContainerStats,
) (row map[string]interface{}) {
	row = make(map[string]interface{})

	// Timestamp
	row[colTimestamp] = stats.Timestamp

	// Machine name
	row[colMachineName] = self.machineName

	// Container name
	name := ref.Name
	if len(ref.Aliases) > 0 {
		name = ref.Aliases[0]
	}
	row[colContainerName] = name

	// Cumulative Cpu Usage
	row[colCpuCumulativeUsage] = stats.Cpu.Usage.Total

	// Cumulative Cpu Usage in system mode
	row[colCpuCumulativeUsageSystem] = stats.Cpu.Usage.System

	// Cumulative Cpu Usage in user mode
	row[colCpuCumulativeUsageUser] = stats.Cpu.Usage.User

	// Memory Usage
	row[colMemoryUsage] = stats.Memory.Usage

	// Working set size
	row[colMemoryWorkingSet] = stats.Memory.WorkingSet

	// container page fault
	row[colMemoryContainerPgfault] = stats.Memory.ContainerData.Pgfault

	// container major page fault
	row[colMemoryContainerPgmajfault] = stats.Memory.ContainerData.Pgmajfault

	// hierarchical page fault
	row[colMemoryHierarchicalPgfault] = stats.Memory.HierarchicalData.Pgfault

	// hierarchical major page fault
	row[colMemoryHierarchicalPgmajfault] = stats.Memory.HierarchicalData.Pgmajfault

	// Network stats.
	row[colRxBytes] = stats.Network.RxBytes
	row[colRxErrors] = stats.Network.RxErrors
	row[colTxBytes] = stats.Network.TxBytes
	row[colTxErrors] = stats.Network.TxErrors

	// TODO(jnagal): Handle per-cpu stats.

	return
}

func (self *bigqueryStorage) containerFilesystemStatsToRows(
	ref info.ContainerReference,
	stats *info.ContainerStats,
) (rows []map[string]interface{}) {
	for _, fsStat := range stats.Filesystem {
		row := make(map[string]interface{}, 0)
		row[colFsDevice] = fsStat.Device
		row[colFsLimit] = fsStat.Limit
		row[colFsUsage] = fsStat.Usage
		rows = append(rows, row)
	}
	return rows
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
	case string:
		return strconv.ParseUint(x, 10, 64)
	}

	return 0, fmt.Errorf("unknown type")
}

func (self *bigqueryStorage) valuesToContainerStats(columns []string, values []interface{}) (*info.ContainerStats, error) {
	stats := &info.ContainerStats{
		Filesystem: make([]info.FsStats, 0),
	}
	var err error
	for i, col := range columns {
		v := values[i]
		switch {
		case col == colTimestamp:
			if t, ok := v.(time.Time); ok {
				stats.Timestamp = t
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
		case col == colRxBytes:
			stats.Network.RxBytes, err = convertToUint64(v)
		case col == colRxErrors:
			stats.Network.RxErrors, err = convertToUint64(v)
		case col == colTxBytes:
			stats.Network.TxBytes, err = convertToUint64(v)
		case col == colTxErrors:
			stats.Network.TxErrors, err = convertToUint64(v)
		case col == colFsDevice:
			device, ok := v.(string)
			if !ok {
				return nil, fmt.Errorf("filesystem name field is not a string: %+v", v)
			}
			if len(stats.Filesystem) == 0 {
				stats.Filesystem = append(stats.Filesystem, info.FsStats{Device: device})
			} else {
				stats.Filesystem[0].Device = device
			}
		case col == colFsLimit:
			limit, err := convertToUint64(v)
			if err != nil {
				return nil, fmt.Errorf("filesystem limit field %+v invalid: %s", v, err)
			}
			if len(stats.Filesystem) == 0 {
				stats.Filesystem = append(stats.Filesystem, info.FsStats{Limit: limit})
			} else {
				stats.Filesystem[0].Limit = limit
			}
		case col == colFsUsage:
			usage, err := convertToUint64(v)
			if err != nil {
				return nil, fmt.Errorf("filesystem usage field %+v invalid: %s", v, err)
			}
			if len(stats.Filesystem) == 0 {
				stats.Filesystem = append(stats.Filesystem, info.FsStats{Usage: usage})
			} else {
				stats.Filesystem[0].Usage = usage
			}
		}
		if err != nil {
			return nil, fmt.Errorf("column %v has invalid value %v: %v", col, v, err)
		}
	}
	return stats, nil
}

func (self *bigqueryStorage) AddStats(ref info.ContainerReference, stats *info.ContainerStats) error {
	if stats == nil {
		return nil
	}
	rows := make([]map[string]interface{}, 0)
	rows = append(rows, self.containerStatsToRows(ref, stats))
	rows = append(rows, self.containerFilesystemStatsToRows(ref, stats)...)
	for _, row := range rows {
		err := self.client.InsertRow(row)
		if err != nil {
			return err
		}
	}
	return nil
}

func (self *bigqueryStorage) getRecentRows(containerName string, numRows int) ([]string, [][]interface{}, error) {
	tableName, err := self.client.GetTableName()
	if err != nil {
		return nil, nil, err
	}

	query := fmt.Sprintf("SELECT * FROM %v WHERE %v='%v' and %v='%v'", tableName, colContainerName, containerName, colMachineName, self.machineName)
	if numRows > 0 {
		query = fmt.Sprintf("%v LIMIT %v", query, numRows)
	}

	return self.client.Query(query)
}

func (self *bigqueryStorage) RecentStats(containerName string, numStats int) ([]*info.ContainerStats, error) {
	if numStats == 0 {
		return nil, nil
	}
	header, rows, err := self.getRecentRows(containerName, numStats)
	if err != nil {
		return nil, err
	}
	statsList := make([]*info.ContainerStats, 0, len(rows))
	for _, row := range rows {
		stats, err := self.valuesToContainerStats(header, row)
		if err != nil {
			return nil, err
		}
		if stats == nil {
			continue
		}
		statsList = append(statsList, stats)
	}
	return statsList, nil
}

func (self *bigqueryStorage) Close() error {
	self.client.Close()
	self.client = nil
	return nil
}

// Create a new bigquery storage driver.
// machineName: A unique identifier to identify the host that current cAdvisor
// instance is running on.
// tableName: BigQuery table used for storing stats.
func New(machineName,
	datasetId,
	tableName string,
) (storage.StorageDriver, error) {
	bqClient, err := client.NewClient()
	if err != nil {
		return nil, err
	}
	err = bqClient.CreateDataset(datasetId)
	if err != nil {
		return nil, err
	}

	ret := &bigqueryStorage{
		client:      bqClient,
		machineName: machineName,
	}
	schema := ret.GetSchema()
	err = bqClient.CreateTable(tableName, schema)
	if err != nil {
		return nil, err
	}
	return ret, nil
}
