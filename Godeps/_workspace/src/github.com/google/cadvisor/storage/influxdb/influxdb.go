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
	"net/url"
	"os"
	"sync"
	"time"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/storage"
	"github.com/google/cadvisor/version"

	influxdb "github.com/influxdb/influxdb/client"
)

func init() {
	storage.RegisterStorageDriver("influxdb", new)
}

type influxdbStorage struct {
	client          *influxdb.Client
	machineName     string
	database        string
	retentionPolicy string
	bufferDuration  time.Duration
	lastWrite       time.Time
	points          []*influxdb.Point
	lock            sync.Mutex
	readyToFlush    func() bool
}

// Series names
const (
	// Cumulative CPU usage
	serCpuUsageTotal  string = "cpu_usage_total"
	serCpuUsageSystem string = "cpu_usage_system"
	serCpuUsageUser   string = "cpu_usage_user"
	serCpuUsagePerCpu string = "cpu_usage_per_cpu"
	// Smoothed average of number of runnable threads x 1000.
	serLoadAverage string = "load_average"
	// Memory Usage
	serMemoryUsage string = "memory_usage"
	// Working set size
	serMemoryWorkingSet string = "memory_working_set"
	// Cumulative count of bytes received.
	serRxBytes string = "rx_bytes"
	// Cumulative count of receive errors encountered.
	serRxErrors string = "rx_errors"
	// Cumulative count of bytes transmitted.
	serTxBytes string = "tx_bytes"
	// Cumulative count of transmit errors encountered.
	serTxErrors string = "tx_errors"
	// Filesystem device.
	serFsDevice string = "fs_device"
	// Filesystem limit.
	serFsLimit string = "fs_limit"
	// Filesystem usage.
	serFsUsage string = "fs_usage"
)

func new() (storage.StorageDriver, error) {
	hostname, err := os.Hostname()
	if err != nil {
		return nil, err
	}
	return newStorage(
		hostname,
		*storage.ArgDbTable,
		*storage.ArgDbName,
		*storage.ArgDbUsername,
		*storage.ArgDbPassword,
		*storage.ArgDbHost,
		*storage.ArgDbIsSecure,
		*storage.ArgDbBufferDuration,
	)
}

// Field names
const (
	fieldValue  string = "value"
	fieldType   string = "type"
	fieldDevice string = "device"
)

// Tag names
const (
	tagMachineName   string = "machine"
	tagContainerName string = "container_name"
)

func (self *influxdbStorage) containerFilesystemStatsToPoints(
	ref info.ContainerReference,
	stats *info.ContainerStats) (points []*influxdb.Point) {
	if len(stats.Filesystem) == 0 {
		return points
	}
	for _, fsStat := range stats.Filesystem {
		tagsFsUsage := map[string]string{
			fieldDevice: fsStat.Device,
			fieldType:   "usage",
		}
		fieldsFsUsage := map[string]interface{}{
			fieldValue: int64(fsStat.Usage),
		}
		pointFsUsage := &influxdb.Point{
			Measurement: serFsUsage,
			Tags:        tagsFsUsage,
			Fields:      fieldsFsUsage,
		}

		tagsFsLimit := map[string]string{
			fieldDevice: fsStat.Device,
			fieldType:   "limit",
		}
		fieldsFsLimit := map[string]interface{}{
			fieldValue: int64(fsStat.Limit),
		}
		pointFsLimit := &influxdb.Point{
			Measurement: serFsLimit,
			Tags:        tagsFsLimit,
			Fields:      fieldsFsLimit,
		}

		points = append(points, pointFsUsage, pointFsLimit)
	}

	self.tagPoints(ref, stats, points)

	return points
}

// Set tags and timestamp for all points of the batch.
// Points should inherit the tags that are set for BatchPoints, but that does not seem to work.
func (self *influxdbStorage) tagPoints(ref info.ContainerReference, stats *info.ContainerStats, points []*influxdb.Point) {
	// Use container alias if possible
	var containerName string
	if len(ref.Aliases) > 0 {
		containerName = ref.Aliases[0]
	} else {
		containerName = ref.Name
	}

	commonTags := map[string]string{
		tagMachineName:   self.machineName,
		tagContainerName: containerName,
	}
	for i := 0; i < len(points); i++ {
		// merge with existing tags if any
		addTagsToPoint(points[i], commonTags)
		points[i].Time = stats.Timestamp
	}
}

func (self *influxdbStorage) containerStatsToPoints(
	ref info.ContainerReference,
	stats *info.ContainerStats,
) (points []*influxdb.Point) {
	// CPU usage: Total usage in nanoseconds
	points = append(points, makePoint(serCpuUsageTotal, stats.Cpu.Usage.Total))

	// CPU usage: Time spend in system space (in nanoseconds)
	points = append(points, makePoint(serCpuUsageSystem, stats.Cpu.Usage.System))

	// CPU usage: Time spent in user space (in nanoseconds)
	points = append(points, makePoint(serCpuUsageUser, stats.Cpu.Usage.User))

	// CPU usage per CPU
	for i := 0; i < len(stats.Cpu.Usage.PerCpu); i++ {
		point := makePoint(serCpuUsagePerCpu, stats.Cpu.Usage.PerCpu[i])
		tags := map[string]string{"instance": fmt.Sprintf("%v", i)}
		addTagsToPoint(point, tags)

		points = append(points, point)
	}

	// Load Average
	points = append(points, makePoint(serLoadAverage, stats.Cpu.LoadAverage))

	// Memory Usage
	points = append(points, makePoint(serMemoryUsage, stats.Memory.Usage))

	// Working Set Size
	points = append(points, makePoint(serMemoryWorkingSet, stats.Memory.WorkingSet))

	// Network Stats
	points = append(points, makePoint(serRxBytes, stats.Network.RxBytes))
	points = append(points, makePoint(serRxErrors, stats.Network.RxErrors))
	points = append(points, makePoint(serTxBytes, stats.Network.TxBytes))
	points = append(points, makePoint(serTxErrors, stats.Network.TxErrors))

	self.tagPoints(ref, stats, points)

	return points
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
	var pointsToFlush []*influxdb.Point
	func() {
		// AddStats will be invoked simultaneously from multiple threads and only one of them will perform a write.
		self.lock.Lock()
		defer self.lock.Unlock()

		self.points = append(self.points, self.containerStatsToPoints(ref, stats)...)
		self.points = append(self.points, self.containerFilesystemStatsToPoints(ref, stats)...)
		if self.readyToFlush() {
			pointsToFlush = self.points
			self.points = make([]*influxdb.Point, 0)
			self.lastWrite = time.Now()
		}
	}()
	if len(pointsToFlush) > 0 {
		points := make([]influxdb.Point, len(pointsToFlush))
		for i, p := range pointsToFlush {
			points[i] = *p
		}

		batchTags := map[string]string{tagMachineName: self.machineName}
		bp := influxdb.BatchPoints{
			Points:   points,
			Database: self.database,
			Tags:     batchTags,
			Time:     stats.Timestamp,
		}
		response, err := self.client.Write(bp)
		if err != nil || checkResponseForErrors(response) != nil {
			return fmt.Errorf("failed to write stats to influxDb - %s", err)
		}
	}
	return nil
}

func (self *influxdbStorage) Close() error {
	self.client = nil
	return nil
}

// machineName: A unique identifier to identify the host that current cAdvisor
// instance is running on.
// influxdbHost: The host which runs influxdb (host:port)
func newStorage(
	machineName,
	tablename,
	database,
	username,
	password,
	influxdbHost string,
	isSecure bool,
	bufferDuration time.Duration,
) (*influxdbStorage, error) {
	url := &url.URL{
		Scheme: "http",
		Host:   influxdbHost,
	}
	if isSecure {
		url.Scheme = "https"
	}

	config := &influxdb.Config{
		URL:       *url,
		Username:  username,
		Password:  password,
		UserAgent: fmt.Sprintf("%v/%v", "cAdvisor", version.Info["version"]),
	}
	client, err := influxdb.NewClient(*config)
	if err != nil {
		return nil, err
	}

	ret := &influxdbStorage{
		client:         client,
		machineName:    machineName,
		database:       database,
		bufferDuration: bufferDuration,
		lastWrite:      time.Now(),
		points:         make([]*influxdb.Point, 0),
	}
	ret.readyToFlush = ret.defaultReadyToFlush
	return ret, nil
}

// Creates a measurement point with a single value field
func makePoint(name string, value interface{}) *influxdb.Point {
	fields := map[string]interface{}{
		fieldValue: toSignedIfUnsigned(value),
	}

	return &influxdb.Point{
		Measurement: name,
		Fields:      fields,
	}
}

// Adds additional tags to the existing tags of a point
func addTagsToPoint(point *influxdb.Point, tags map[string]string) {
	if point.Tags == nil {
		point.Tags = tags
	} else {
		for k, v := range tags {
			point.Tags[k] = v
		}
	}
}

// Checks response for possible errors
func checkResponseForErrors(response *influxdb.Response) error {
	const msg = "failed to write stats to influxDb - %s"

	if response != nil && response.Err != nil {
		return fmt.Errorf(msg, response.Err)
	}
	if response != nil && response.Results != nil {
		for _, result := range response.Results {
			if result.Err != nil {
				return fmt.Errorf(msg, result.Err)
			}
			if result.Series != nil {
				for _, row := range result.Series {
					if row.Err != nil {
						return fmt.Errorf(msg, row.Err)
					}
				}
			}
		}
	}
	return nil
}

// Some stats have type unsigned integer, but the InfluxDB client accepts only signed integers.
func toSignedIfUnsigned(value interface{}) interface{} {
	switch v := value.(type) {
	case uint64:
		return int64(v)
	case uint32:
		return int32(v)
	case uint16:
		return int16(v)
	case uint8:
		return int8(v)
	case uint:
		return int(v)
	}
	return value
}
