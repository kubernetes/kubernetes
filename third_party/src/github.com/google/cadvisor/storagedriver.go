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

package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/google/cadvisor/storage"
	"github.com/google/cadvisor/storage/influxdb"
	"github.com/google/cadvisor/storage/memory"
)

var argSampleSize = flag.Int("samples", 1024, "number of samples we want to keep")
var argHistoryDuration = flag.Int("history_duration", 60, "number of seconds of container history to keep")
var argDbUsername = flag.String("storage_driver_user", "root", "database username")
var argDbPassword = flag.String("storage_driver_password", "root", "database password")
var argDbHost = flag.String("storage_driver_host", "localhost:8086", "database host:port")
var argDbName = flag.String("storage_driver_name", "cadvisor", "database name")
var argDbIsSecure = flag.Bool("storage_driver_secure", false, "use secure connection with database")

func NewStorageDriver(driverName string) (storage.StorageDriver, error) {
	var storageDriver storage.StorageDriver
	var err error
	switch driverName {
	case "":
		// empty string by default is the in memory store
		fallthrough
	case "memory":
		storageDriver = memory.New(*argSampleSize, *argHistoryDuration)
		return storageDriver, nil
	case "influxdb":
		var hostname string
		hostname, err = os.Hostname()
		if err != nil {
			return nil, err
		}

		storageDriver, err = influxdb.New(
			hostname,
			"cadvisorTable",
			*argDbName,
			*argDbUsername,
			*argDbPassword,
			*argDbHost,
			*argDbIsSecure,
			// TODO(monnand): One hour? Or user-defined?
			1*time.Hour,
		)
	default:
		err = fmt.Errorf("Unknown database driver: %v", *argDbDriver)
	}
	if err != nil {
		return nil, err
	}
	return storageDriver, nil
}
