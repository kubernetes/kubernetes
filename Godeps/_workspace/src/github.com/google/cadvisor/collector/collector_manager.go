// Copyright 2015 Google Inc. All Rights Reserved.
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

package collector

import (
	"fmt"
	"strings"
	"time"

	"github.com/google/cadvisor/info/v2"
)

type collectorManager struct {
	collectors []*collectorData
}

var _ CollectorManager = &collectorManager{}

type collectorData struct {
	collector          Collector
	nextCollectionTime time.Time
}

// Returns a new CollectorManager that is thread-compatible.
func NewCollectorManager() (CollectorManager, error) {
	return &collectorManager{
		collectors: []*collectorData{},
	}, nil
}

func (cm *collectorManager) RegisterCollector(collector Collector) error {
	cm.collectors = append(cm.collectors, &collectorData{
		collector:          collector,
		nextCollectionTime: time.Now(),
	})
	return nil
}

func (cm *collectorManager) Collect() (time.Time, []v2.Metric, error) {
	var errors []error

	// Collect from all collectors that are ready.
	var next time.Time
	var metrics []v2.Metric
	for _, c := range cm.collectors {
		if c.nextCollectionTime.Before(time.Now()) {
			nextCollection, newMetrics, err := c.collector.Collect()
			if err != nil {
				errors = append(errors, err)
			}
			metrics = append(metrics, newMetrics...)
			c.nextCollectionTime = nextCollection
		}

		// Keep track of the next collector that will be ready.
		if next.IsZero() || next.After(c.nextCollectionTime) {
			next = c.nextCollectionTime
		}
	}

	return next, metrics, compileErrors(errors)
}

// Make an error slice into a single error.
func compileErrors(errors []error) error {
	if len(errors) == 0 {
		return nil
	}

	res := make([]string, len(errors))
	for i := range errors {
		res[i] = fmt.Sprintf("Error %d: %v", i, errors[i].Error())
	}
	return fmt.Errorf("%s", strings.Join(res, ","))
}
