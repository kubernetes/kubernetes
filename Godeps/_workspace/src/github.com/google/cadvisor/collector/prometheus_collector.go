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
	"io/ioutil"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/google/cadvisor/info/v1"
)

type PrometheusCollector struct {
	//name of the collector
	name string

	//rate at which metrics are collected
	pollingFrequency time.Duration

	//holds information extracted from the config file for a collector
	configFile Prometheus
}

//Returns a new collector using the information extracted from the configfile
func NewPrometheusCollector(collectorName string, configFile []byte) (*PrometheusCollector, error) {
	var configInJSON Prometheus
	err := json.Unmarshal(configFile, &configInJSON)
	if err != nil {
		return nil, err
	}

	minPollingFrequency := configInJSON.PollingFrequency

	// Minimum supported frequency is 1s
	minSupportedFrequency := 1 * time.Second

	if minPollingFrequency < minSupportedFrequency {
		minPollingFrequency = minSupportedFrequency
	}

	//TODO : Add checks for validity of config file (eg : Accurate JSON fields)
	return &PrometheusCollector{
		name:             collectorName,
		pollingFrequency: minPollingFrequency,
		configFile:       configInJSON,
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
	response, err := http.Get(collector.configFile.Endpoint)
	if err != nil {
		return specs
	}
	defer response.Body.Close()

	pageContent, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return specs
	}

	lines := strings.Split(string(pageContent), "\n")
	for i, line := range lines {
		if strings.HasPrefix(line, "# HELP") {
			stopIndex := strings.Index(lines[i+2], "{")
			if stopIndex == -1 {
				stopIndex = strings.Index(lines[i+2], " ")
			}
			spec := v1.MetricSpec{
				Name:   strings.TrimSpace(lines[i+2][0:stopIndex]),
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

	uri := collector.configFile.Endpoint
	response, err := http.Get(uri)
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
			metrics[metName] = append(metrics[metName], metric)
		}
	}
	return nextCollectionTime, metrics, compileErrors(errorSlice)
}
