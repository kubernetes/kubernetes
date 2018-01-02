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
	"strings"
	"time"

	"github.com/google/cadvisor/cache/memory"
	"github.com/google/cadvisor/storage"
	_ "github.com/google/cadvisor/storage/bigquery"
	_ "github.com/google/cadvisor/storage/elasticsearch"
	_ "github.com/google/cadvisor/storage/influxdb"
	_ "github.com/google/cadvisor/storage/kafka"
	_ "github.com/google/cadvisor/storage/redis"
	_ "github.com/google/cadvisor/storage/statsd"
	_ "github.com/google/cadvisor/storage/stdout"

	"github.com/golang/glog"
)

var (
	storageDriver   = flag.String("storage_driver", "", fmt.Sprintf("Storage `driver` to use. Data is always cached shortly in memory, this controls where data is pushed besides the local cache. Empty means none. Options are: <empty>, %s", strings.Join(storage.ListDrivers(), ", ")))
	storageDuration = flag.Duration("storage_duration", 2*time.Minute, "How long to keep data stored (Default: 2min).")
)

// NewMemoryStorage creates a memory storage with an optional backend storage option.
func NewMemoryStorage() (*memory.InMemoryCache, error) {
	backendStorage, err := storage.New(*storageDriver)
	if err != nil {
		return nil, err
	}
	if *storageDriver != "" {
		glog.V(1).Infof("Using backend storage type %q", *storageDriver)
	}
	glog.V(1).Infof("Caching stats in memory for %v", *storageDuration)
	return memory.New(*storageDuration, backendStorage), nil
}
