// Copyright 2016 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package push_test

import (
	"fmt"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/push"
)

var (
	completionTime = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "db_backup_last_completion_timestamp_seconds",
		Help: "The timestamp of the last completion of a DB backup, successful or not.",
	})
	successTime = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "db_backup_last_success_timestamp_seconds",
		Help: "The timestamp of the last successful completion of a DB backup.",
	})
	duration = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "db_backup_duration_seconds",
		Help: "The duration of the last DB backup in seconds.",
	})
	records = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "db_backup_records_processed",
		Help: "The number of records processed in the last DB backup.",
	})
)

func performBackup() (int, error) {
	// Perform the backup and return the number of backed up records and any
	// applicable error.
	// ...
	return 42, nil
}

func ExamplePusher_Add() {
	// We use a registry here to benefit from the consistency checks that
	// happen during registration.
	registry := prometheus.NewRegistry()
	registry.MustRegister(completionTime, duration, records)
	// Note that successTime is not registered.

	pusher := push.New("http://pushgateway:9091", "db_backup").Gatherer(registry)

	start := time.Now()
	n, err := performBackup()
	records.Set(float64(n))
	// Note that time.Since only uses a monotonic clock in Go1.9+.
	duration.Set(time.Since(start).Seconds())
	completionTime.SetToCurrentTime()
	if err != nil {
		fmt.Println("DB backup failed:", err)
	} else {
		// Add successTime to pusher only in case of success.
		// We could as well register it with the registry.
		// This example, however, demonstrates that you can
		// mix Gatherers and Collectors when handling a Pusher.
		pusher.Collector(successTime)
		successTime.SetToCurrentTime()
	}
	// Add is used here rather than Push to not delete a previously pushed
	// success timestamp in case of a failure of this backup.
	if err := pusher.Add(); err != nil {
		fmt.Println("Could not push to Pushgateway:", err)
	}
}
