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
	"testing"
	"time"

	"github.com/google/cadvisor/storage"
	"github.com/google/cadvisor/storage/test"
	"github.com/influxdb/influxdb-go"
)

func runStorageTest(f func(storage.StorageDriver, *testing.T), t *testing.T) {
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
		percentilesDuration)
	if err != nil {
		t.Fatal(err)
	}
	// generate another container's data on same machine.
	test.StorageDriverFillRandomStatsFunc("containerOnSameMachine", 100, driver, t)

	// generate another container's data on another machine.
	driverForAnotherMachine, err := New("machineB",
		tablename,
		database,
		username,
		password,
		hostname,
		false,
		percentilesDuration)
	if err != nil {
		t.Fatal(err)
	}
	defer driverForAnotherMachine.Close()
	test.StorageDriverFillRandomStatsFunc("containerOnAnotherMachine", 100, driverForAnotherMachine, t)
	f(driver, t)
}

func TestSampleCpuUsage(t *testing.T) {
	runStorageTest(test.StorageDriverTestSampleCpuUsage, t)
}

func TestRetrievePartialRecentStats(t *testing.T) {
	runStorageTest(test.StorageDriverTestRetrievePartialRecentStats, t)
}

func TestSamplesWithoutSample(t *testing.T) {
	runStorageTest(test.StorageDriverTestSamplesWithoutSample, t)
}

func TestRetrieveAllRecentStats(t *testing.T) {
	runStorageTest(test.StorageDriverTestRetrieveAllRecentStats, t)
}

func TestNoRecentStats(t *testing.T) {
	runStorageTest(test.StorageDriverTestNoRecentStats, t)
}

func TestNoSamples(t *testing.T) {
	runStorageTest(test.StorageDriverTestNoSamples, t)
}

func TestPercentiles(t *testing.T) {
	runStorageTest(test.StorageDriverTestPercentiles, t)
}

func TestMaxMemoryUsage(t *testing.T) {
	runStorageTest(test.StorageDriverTestMaxMemoryUsage, t)
}

func TestPercentilesWithoutSample(t *testing.T) {
	runStorageTest(test.StorageDriverTestPercentilesWithoutSample, t)
}

func TestPercentilesWithoutStats(t *testing.T) {
	runStorageTest(test.StorageDriverTestPercentilesWithoutStats, t)
}
