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
	"time"

	"github.com/google/cadvisor/info/v1"
)

type Config struct {
	//the endpoint to hit to scrape metrics
	Endpoint string `json:"endpoint"`

	//holds information about different metrics that can be collected
	MetricsConfig []MetricConfig `json:"metrics_config"`
}

// metricConfig holds information extracted from the config file about a metric
type MetricConfig struct {
	//the name of the metric
	Name string `json:"name"`

	//enum type for the metric type
	MetricType v1.MetricType `json:"metric_type"`

	// metric units to display on UI and in storage (eg: MB, cores)
	// this is only used for display.
	Units string `json:"units"`

	//data type of the metric (eg: int, float)
	DataType v1.DataType `json:"data_type"`

	//the frequency at which the metric should be collected
	PollingFrequency time.Duration `json:"polling_frequency"`

	//the regular expression that can be used to extract the metric
	Regex string `json:"regex"`
}

type Prometheus struct {
	//the endpoint to hit to scrape metrics
	Endpoint string `json:"endpoint"`

	//the frequency at which metrics should be collected
	PollingFrequency time.Duration `json:"polling_frequency"`

	//holds names of different metrics that can be collected
	MetricsConfig []string `json:"metrics_config"`
}
