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

// +build influxdb_test
// To run unit test: go test -tags influxdb_test

package influxdb

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/storage"
	"github.com/google/cadvisor/storage/test"
	influxdb "github.com/influxdb/influxdb/client"
)

// The duration in seconds for which stats will be buffered in the influxdb driver.
const kCacheDuration = 1

type influxDbTestStorageDriver struct {
	count  int
	buffer int
	base   storage.StorageDriver
}

func (self *influxDbTestStorageDriver) readyToFlush() bool {
	if self.count >= self.buffer {
		return true
	}
	return false
}

func (self *influxDbTestStorageDriver) AddStats(ref info.ContainerReference, stats *info.ContainerStats) error {
	self.count++
	return self.base.AddStats(ref, stats)
}

func (self *influxDbTestStorageDriver) RecentStats(containerName string, numStats int) ([]*info.ContainerStats, error) {
	return nil, nil
}

func (self *influxDbTestStorageDriver) Percentiles(containerName string, cpuUsagePercentiles []int, memUsagePercentiles []int) (*info.ContainerStatsPercentiles, error) {
	return self.base.Percentiles(containerName, cpuUsagePercentiles, memUsagePercentiles)
}

func (self *influxDbTestStorageDriver) Samples(containerName string, numSamples int) ([]*info.ContainerStatsSample, error) {
	return self.base.Samples(containerName, numSamples)
}

func (self *influxDbTestStorageDriver) Close() error {
	return self.base.Close()
}

func (self *influxDbTestStorageDriver) StatsEq(a, b *info.ContainerStats) bool {
	if !test.TimeEq(a.Timestamp, b.Timestamp, 10*time.Millisecond) {
		return false
	}
	// Check only the stats populated in influxdb.
	if a.Cpu.Usage.Total != b.Cpu.Usage.Total {
		return false
	}

	if a.Memory.Usage != b.Memory.Usage {
		return false
	}

	if a.Memory.WorkingSet != b.Memory.WorkingSet {
		return false
	}

	if !reflect.DeepEqual(a.Network, b.Network) {
		return false
	}

	if !reflect.DeepEqual(a.Filesystem, b.Filesystem) {
		return false
	}
	return true
}

func runStorageTest(f func(test.TestStorageDriver, *testing.T), t *testing.T, bufferCount int) {
	machineName := "machineA"
	tablename := "t"
	database := "cadvisor"
	username := "root"
	password := "root"
	hostname := "localhost:8086"
	percentilesDuration := 10 * time.Minute
	rootConfig := &influxdb.ClientConfig{
		Host:     hostname,
		Username: username,
		Password: password,
		IsSecure: false,
	}
	rootClient, err := influxdb.NewClient(rootConfig)
	if err != nil {
		t.Fatal(err)
	}
	// create the data base first.
	rootClient.CreateDatabase(database)
	config := &influxdb.ClientConfig{
		Host:     hostname,
		Username: username,
		Password: password,
		Database: database,
		IsSecure: false,
	}
	client, err := influxdb.NewClient(config)
	if err != nil {
		t.Fatal(err)
	}
	client.DisableCompression()
	deleteAll := fmt.Sprintf("drop series %v", tablename)
	_, err = client.Query(deleteAll)
	if err != nil {
		t.Fatal(err)
	}
	// delete all data by the end of the call
	defer client.Query(deleteAll)

	driver, err := New(machineName,
		tablename,
		database,
		username,
		password,
		hostname,
		false,
		time.Duration(bufferCount),
		percentilesDuration)
	if err != nil {
		t.Fatal(err)
	}
	testDriver := &influxDbTestStorageDriver{buffer: bufferCount}
	driver.OverrideReadyToFlush(testDriver.readyToFlush)
	testDriver.base = driver

	// generate another container's data on same machine.
	test.StorageDriverFillRandomStatsFunc("containerOnSameMachine", 100, testDriver, t)

	// generate another container's data on another machine.
	driverForAnotherMachine, err := New("machineB",
		tablename,
		database,
		username,
		password,
		hostname,
		false,
		time.Duration(bufferCount),
		percentilesDuration)
	if err != nil {
		t.Fatal(err)
	}
	defer driverForAnotherMachine.Close()
	testDriverOtherMachine := &influxDbTestStorageDriver{buffer: bufferCount}
	driverForAnotherMachine.OverrideReadyToFlush(testDriverOtherMachine.readyToFlush)
	testDriverOtherMachine.base = driverForAnotherMachine

	test.StorageDriverFillRandomStatsFunc("containerOnAnotherMachine", 100, testDriverOtherMachine, t)
	f(testDriver, t)
}

func TestRetrievePartialRecentStats(t *testing.T) {
	runStorageTest(test.StorageDriverTestRetrievePartialRecentStats, t, 20)
}

func TestRetrieveAllRecentStats(t *testing.T) {
	runStorageTest(test.StorageDriverTestRetrieveAllRecentStats, t, 10)
}

func TestNoRecentStats(t *testing.T) {
	runStorageTest(test.StorageDriverTestNoRecentStats, t, kCacheDuration)
}
