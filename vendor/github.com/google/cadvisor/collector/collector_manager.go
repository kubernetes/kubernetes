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

	"github.com/google/cadvisor/info/v1"
)

const metricLabelPrefix = "io.cadvisor.metric."

type GenericCollectorManager struct {
	Collectors         []*collectorData
	NextCollectionTime time.Time
}

type collectorData struct {
	collector          Collector
	nextCollectionTime time.Time
}

// Returns a new CollectorManager that is thread-compatible.
func NewCollectorManager() (CollectorManager, error) {
	return &GenericCollectorManager{
		Collectors:         []*collectorData{},
		NextCollectionTime: time.Now(),
	}, nil
}

func GetCollectorConfigs(labels map[string]string) map[string]string {
	configs := map[string]string{}
	for k, v := range labels {
		if strings.HasPrefix(k, metricLabelPrefix) {
			name := strings.TrimPrefix(k, metricLabelPrefix)
			configs[name] = v
		}
	}
	return configs
}

func (cm *GenericCollectorManager) RegisterCollector(collector Collector) error {
	cm.Collectors = append(cm.Collectors, &collectorData{
		collector:          collector,
		nextCollectionTime: time.Now(),
	})
	return nil
}

func (cm *GenericCollectorManager) GetSpec() ([]v1.MetricSpec, error) {
	metricSpec := []v1.MetricSpec{}
	for _, c := range cm.Collectors {
		specs := c.collector.GetSpec()
		metricSpec = append(metricSpec, specs...)
	}

	return metricSpec, nil
}

func (cm *GenericCollectorManager) Collect() (time.Time, map[string][]v1.MetricVal, error) {
	var errors []error

	// Collect from all collectors that are ready.
	var next time.Time
	metrics := map[string][]v1.MetricVal{}
	for _, c := range cm.Collectors {
		if c.nextCollectionTime.Before(time.Now()) {
			var err error
			c.nextCollectionTime, metrics, err = c.collector.Collect(metrics)
			if err != nil {
				errors = append(errors, err)
			}
		}

		// Keep track of the next collector that will be ready.
		if next.IsZero() || next.After(c.nextCollectionTime) {
			next = c.nextCollectionTime
		}
	}
	cm.NextCollectionTime = next
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
