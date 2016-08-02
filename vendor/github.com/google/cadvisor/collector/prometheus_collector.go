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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/info/v1"
)

type PrometheusCollector struct {
	//name of the collector
	name string

	//rate at which metrics are collected
	pollingFrequency time.Duration

	//holds information extracted from the config file for a collector
	configFile Prometheus

	// the metrics to gather (uses a map as a set)
	metricsSet map[string]bool

	// Limit for the number of scaped metrics. If the count is higher,
	// no metrics will be returned.
	metricCountLimit int

	// The Http client to use when connecting to metric endpoints
	httpClient *http.Client
}

//Returns a new collector using the information extracted from the configfile
func NewPrometheusCollector(collectorName string, configFile []byte, metricCountLimit int, containerHandler container.ContainerHandler, httpClient *http.Client) (*PrometheusCollector, error) {
	var configInJSON Prometheus
	err := json.Unmarshal(configFile, &configInJSON)
	if err != nil {
		return nil, err
	}

	configInJSON.Endpoint.configure(containerHandler)

	minPollingFrequency := configInJSON.PollingFrequency

	// Minimum supported frequency is 1s
	minSupportedFrequency := 1 * time.Second

	if minPollingFrequency < minSupportedFrequency {
		minPollingFrequency = minSupportedFrequency
	}

	if metricCountLimit < 0 {
		return nil, fmt.Errorf("Metric count limit must be greater than 0")
	}

	var metricsSet map[string]bool
	if len(configInJSON.MetricsConfig) > 0 {
		metricsSet = make(map[string]bool, len(configInJSON.MetricsConfig))
		for _, name := range configInJSON.MetricsConfig {
			metricsSet[name] = true
		}
	}

	if len(configInJSON.MetricsConfig) > metricCountLimit {
		return nil, fmt.Errorf("Too many metrics defined: %d limit %d", len(configInJSON.MetricsConfig), metricCountLimit)
	}

	//TODO : Add checks for validity of config file (eg : Accurate JSON fields)
	return &PrometheusCollector{
		name:             collectorName,
		pollingFrequency: minPollingFrequency,
		configFile:       configInJSON,
		metricsSet:       metricsSet,
		metricCountLimit: metricCountLimit,
		httpClient:       httpClient,
	}, nil
}

//Returns name of the collector
func (collector *PrometheusCollector) Name() string {
	return collector.name
}

func getMetricData(line string) string {
	fields := strings.Fields(line)
	data := fields[3]
	if len(fields) > 4 {
		for i := range fields {
			if i > 3 {
				data = data + "_" + fields[i]
			}
		}
	}
	return strings.TrimSpace(data)
}

func (collector *PrometheusCollector) GetSpec() []v1.MetricSpec {
	specs := []v1.MetricSpec{}

	response, err := collector.httpClient.Get(collector.configFile.Endpoint.URL)
	if err != nil {
		return specs
	}
	defer response.Body.Close()

	pageContent, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return specs
	}

	lines := strings.Split(string(pageContent), "\n")
	lineCount := len(lines)
	for i, line := range lines {
		if strings.HasPrefix(line, "# HELP") {
			if i+2 >= lineCount {
				break
			}

			stopIndex := strings.IndexAny(lines[i+2], "{ ")
			if stopIndex == -1 {
				continue
			}

			name := strings.TrimSpace(lines[i+2][0:stopIndex])
			if _, ok := collector.metricsSet[name]; collector.metricsSet != nil && !ok {
				continue
			}
			spec := v1.MetricSpec{
				Name:   name,
				Type:   v1.MetricType(getMetricData(lines[i+1])),
				Format: "float",
				Units:  getMetricData(lines[i]),
			}
			specs = append(specs, spec)
		}
	}
	return specs
}

//Returns collected metrics and the next collection time of the collector
func (collector *PrometheusCollector) Collect(metrics map[string][]v1.MetricVal) (time.Time, map[string][]v1.MetricVal, error) {
	currentTime := time.Now()
	nextCollectionTime := currentTime.Add(time.Duration(collector.pollingFrequency))

	uri := collector.configFile.Endpoint.URL
	response, err := collector.httpClient.Get(uri)
	if err != nil {
		return nextCollectionTime, nil, err
	}
	defer response.Body.Close()

	pageContent, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nextCollectionTime, nil, err
	}

	var errorSlice []error
	lines := strings.Split(string(pageContent), "\n")

	newMetrics := make(map[string][]v1.MetricVal)

	for _, line := range lines {
		if line == "" {
			break
		}
		if !strings.HasPrefix(line, "# HELP") && !strings.HasPrefix(line, "# TYPE") {
			var metLabel string
			startLabelIndex := strings.Index(line, "{")
			spaceIndex := strings.Index(line, " ")
			if startLabelIndex == -1 {
				startLabelIndex = spaceIndex
			}

			metName := strings.TrimSpace(line[0:startLabelIndex])
			if _, ok := collector.metricsSet[metName]; collector.metricsSet != nil && !ok {
				continue
			}

			if startLabelIndex+1 <= spaceIndex-1 {
				metLabel = strings.TrimSpace(line[(startLabelIndex + 1):(spaceIndex - 1)])
			}

			metVal, err := strconv.ParseFloat(line[spaceIndex+1:], 64)
			if err != nil {
				errorSlice = append(errorSlice, err)
			}
			if math.IsNaN(metVal) {
				metVal = 0
			}

			metric := v1.MetricVal{
				Label:      metLabel,
				FloatValue: metVal,
				Timestamp:  currentTime,
			}
			newMetrics[metName] = append(newMetrics[metName], metric)
			if len(newMetrics) > collector.metricCountLimit {
				return nextCollectionTime, nil, fmt.Errorf("too many metrics to collect")
			}
		}
	}
	for key, val := range newMetrics {
		metrics[key] = append(metrics[key], val...)
	}

	return nextCollectionTime, metrics, compileErrors(errorSlice)
}
