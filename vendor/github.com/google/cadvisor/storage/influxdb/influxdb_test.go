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
	"math/rand"
	"net/url"
	"reflect"
	"testing"
	"time"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/storage"
	"github.com/google/cadvisor/storage/test"

	influxdb "github.com/influxdb/influxdb/client"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The duration in seconds for which stats will be buffered in the influxdb driver.
const kCacheDuration = 1

type influxDbTestStorageDriver struct {
	count  int
	buffer int
	base   storage.StorageDriver
}

func (self *influxDbTestStorageDriver) readyToFlush() bool {
	return self.count >= self.buffer
}

func (self *influxDbTestStorageDriver) AddStats(ref info.ContainerReference, stats *info.ContainerStats) error {
	self.count++
	return self.base.AddStats(ref, stats)
}

func (self *influxDbTestStorageDriver) Close() error {
	return self.base.Close()
}

func (self *influxDbTestStorageDriver) StatsEq(a, b *info.ContainerStats) bool {
	if !test.TimeEq(a.Timestamp, b.Timestamp, 10*time.Millisecond) {
		return false
	}
	// Check only the stats populated in influxdb.
	if !reflect.DeepEqual(a.Cpu.Usage, b.Cpu.Usage) {
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
	table := "cadvisor_table"
	database := "cadvisor_test"
	username := "root"
	password := "root"
	hostname := "localhost:8086"
	//percentilesDuration := 10 * time.Minute

	config := influxdb.Config{
		URL:      url.URL{Scheme: "http", Host: hostname},
		Username: username,
		Password: password,
	}
	client, err := influxdb.NewClient(config)
	if err != nil {
		t.Fatal(err)
	}

	// Re-create the database first.
	if err := prepareDatabase(client, database); err != nil {
		t.Fatal(err)
	}

	// Delete all data by the end of the call.
	//defer client.Query(influxdb.Query{Command: fmt.Sprintf("drop database \"%v\"", database)})

	driver, err := newStorage(machineName,
		table,
		database,
		username,
		password,
		hostname,
		false,
		time.Duration(bufferCount))
	if err != nil {
		t.Fatal(err)
	}
	defer driver.Close()
	testDriver := &influxDbTestStorageDriver{buffer: bufferCount}
	driver.OverrideReadyToFlush(testDriver.readyToFlush)
	testDriver.base = driver

	// Generate another container's data on same machine.
	test.StorageDriverFillRandomStatsFunc("containerOnSameMachine", 100, testDriver, t)

	// Generate another container's data on another machine.
	driverForAnotherMachine, err := newStorage("machineB",
		table,
		database,
		username,
		password,
		hostname,
		false,
		time.Duration(bufferCount))
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

func prepareDatabase(client *influxdb.Client, database string) error {
	dropDbQuery := influxdb.Query{
		Command: fmt.Sprintf("drop database \"%v\"", database),
	}
	createDbQuery := influxdb.Query{
		Command: fmt.Sprintf("create database \"%v\"", database),
	}
	// A default retention policy must always be present.
	// Depending on the InfluxDB configuration it may be created automatically with the database or not.
	// TODO create ret. policy only if not present
	createPolicyQuery := influxdb.Query{
		Command: fmt.Sprintf("create retention policy \"default\" on \"%v\" duration 1h replication 1 default", database),
	}
	_, err := client.Query(dropDbQuery)
	if err != nil {
		return err
	}
	_, err = client.Query(createDbQuery)
	if err != nil {
		return err
	}
	_, err = client.Query(createPolicyQuery)
	return err
}

func TestContainerFileSystemStatsToPoints(t *testing.T) {
	assert := assert.New(t)

	machineName := "testMachine"
	table := "cadvisor_table"
	database := "cadvisor_test"
	username := "root"
	password := "root"
	influxdbHost := "localhost:8086"

	storage, err := newStorage(machineName,
		table,
		database,
		username,
		password,
		influxdbHost,
		false, 2*time.Minute)
	assert.Nil(err)

	ref := info.ContainerReference{
		Name: "containerName",
	}
	stats := &info.ContainerStats{}
	points := storage.containerFilesystemStatsToPoints(ref, stats)

	// stats.Filesystem is always nil, not sure why
	assert.Nil(points)
}

func TestContainerStatsToPoints(t *testing.T) {
	// Given
	storage, err := createTestStorage()
	require.Nil(t, err)
	require.NotNil(t, storage)

	ref, stats := createTestStats()
	require.Nil(t, err)
	require.NotNil(t, stats)

	// When
	points := storage.containerStatsToPoints(*ref, stats)

	// Then
	assert.NotEmpty(t, points)
	assert.Len(t, points, 10+len(stats.Cpu.Usage.PerCpu))

	assertContainsPointWithValue(t, points, serCpuUsageTotal, stats.Cpu.Usage.Total)
	assertContainsPointWithValue(t, points, serCpuUsageSystem, stats.Cpu.Usage.System)
	assertContainsPointWithValue(t, points, serCpuUsageUser, stats.Cpu.Usage.User)
	assertContainsPointWithValue(t, points, serMemoryUsage, stats.Memory.Usage)
	assertContainsPointWithValue(t, points, serLoadAverage, stats.Cpu.LoadAverage)
	assertContainsPointWithValue(t, points, serMemoryWorkingSet, stats.Memory.WorkingSet)
	assertContainsPointWithValue(t, points, serRxBytes, stats.Network.RxBytes)
	assertContainsPointWithValue(t, points, serRxErrors, stats.Network.RxErrors)
	assertContainsPointWithValue(t, points, serTxBytes, stats.Network.TxBytes)
	assertContainsPointWithValue(t, points, serTxBytes, stats.Network.TxErrors)

	for _, cpu_usage := range stats.Cpu.Usage.PerCpu {
		assertContainsPointWithValue(t, points, serCpuUsagePerCpu, cpu_usage)
	}
}

func assertContainsPointWithValue(t *testing.T, points []*influxdb.Point, name string, value interface{}) bool {
	found := false
	for _, point := range points {
		if point.Measurement == name && point.Fields[fieldValue] == toSignedIfUnsigned(value) {
			found = true
			break
		}
	}
	return assert.True(t, found, "no point found with name='%v' and value=%v", name, value)
}

func createTestStorage() (*influxdbStorage, error) {
	machineName := "testMachine"
	table := "cadvisor_table"
	database := "cadvisor_test"
	username := "root"
	password := "root"
	influxdbHost := "localhost:8086"

	storage, err := newStorage(machineName,
		table,
		database,
		username,
		password,
		influxdbHost,
		false, 2*time.Minute)

	return storage, err
}

func createTestStats() (*info.ContainerReference, *info.ContainerStats) {
	ref := &info.ContainerReference{
		Name:    "testContainername",
		Aliases: []string{"testContainerAlias1", "testContainerAlias2"},
	}

	cpuUsage := info.CpuUsage{
		Total:  uint64(rand.Intn(10000)),
		PerCpu: []uint64{uint64(rand.Intn(1000)), uint64(rand.Intn(1000)), uint64(rand.Intn(1000))},
		User:   uint64(rand.Intn(10000)),
		System: uint64(rand.Intn(10000)),
	}

	stats := &info.ContainerStats{
		Timestamp: time.Now(),
		Cpu: info.CpuStats{
			Usage:       cpuUsage,
			LoadAverage: int32(rand.Intn(1000)),
		},
	}
	return ref, stats
}
