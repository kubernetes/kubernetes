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
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/google/cadvisor/container"
	v1 "github.com/google/cadvisor/info/v1"
)

type GenericCollector struct {
	// name of the collector
	name string

	// holds information extracted from the config file for a collector
	configFile Config

	// holds information necessary to extract metrics
	info *collectorInfo

	// The Http client to use when connecting to metric endpoints
	httpClient *http.Client
}

type collectorInfo struct {
	// minimum polling frequency among all metrics
	minPollingFrequency time.Duration

	// regular expresssions for all metrics
	regexps []*regexp.Regexp

	// Limit for the number of srcaped metrics. If the count is higher,
	// no metrics will be returned.
	metricCountLimit int
}

// Returns a new collector using the information extracted from the configfile
func NewCollector(collectorName string, configFile []byte, metricCountLimit int, containerHandler container.ContainerHandler, httpClient *http.Client) (*GenericCollector, error) {
	var configInJSON Config
	err := json.Unmarshal(configFile, &configInJSON)
	if err != nil {
		return nil, err
	}

	configInJSON.Endpoint.configure(containerHandler)

	// TODO : Add checks for validity of config file (eg : Accurate JSON fields)

	if len(configInJSON.MetricsConfig) == 0 {
		return nil, fmt.Errorf("No metrics provided in config")
	}

	minPollFrequency := time.Duration(0)
	regexprs := make([]*regexp.Regexp, len(configInJSON.MetricsConfig))

	for ind, metricConfig := range configInJSON.MetricsConfig {
		// Find the minimum specified polling frequency in metric config.
		if metricConfig.PollingFrequency != 0 {
			if minPollFrequency == 0 || metricConfig.PollingFrequency < minPollFrequency {
				minPollFrequency = metricConfig.PollingFrequency
			}
		}

		regexprs[ind], err = regexp.Compile(metricConfig.Regex)
		if err != nil {
			return nil, fmt.Errorf("Invalid regexp %v for metric %v", metricConfig.Regex, metricConfig.Name)
		}
	}

	// Minimum supported polling frequency is 1s.
	minSupportedFrequency := 1 * time.Second
	if minPollFrequency < minSupportedFrequency {
		minPollFrequency = minSupportedFrequency
	}

	if len(configInJSON.MetricsConfig) > metricCountLimit {
		return nil, fmt.Errorf("Too many metrics defined: %d limit: %d", len(configInJSON.MetricsConfig), metricCountLimit)
	}

	return &GenericCollector{
		name:       collectorName,
		configFile: configInJSON,
		info: &collectorInfo{
			minPollingFrequency: minPollFrequency,
			regexps:             regexprs,
			metricCountLimit:    metricCountLimit,
		},
		httpClient: httpClient,
	}, nil
}

// Returns name of the collector
func (collector *GenericCollector) Name() string {
	return collector.name
}

func (collector *GenericCollector) configToSpec(config MetricConfig) v1.MetricSpec {
	return v1.MetricSpec{
		Name:   config.Name,
		Type:   config.MetricType,
		Format: config.DataType,
		Units:  config.Units,
	}
}

func (collector *GenericCollector) GetSpec() []v1.MetricSpec {
	specs := []v1.MetricSpec{}
	for _, metricConfig := range collector.configFile.MetricsConfig {
		spec := collector.configToSpec(metricConfig)
		specs = append(specs, spec)
	}
	return specs
}

// Returns collected metrics and the next collection time of the collector
func (collector *GenericCollector) Collect(metrics map[string][]v1.MetricVal) (time.Time, map[string][]v1.MetricVal, error) {
	currentTime := time.Now()
	nextCollectionTime := currentTime.Add(time.Duration(collector.info.minPollingFrequency))

	uri := collector.configFile.Endpoint.URL
	response, err := collector.httpClient.Get(uri)
	if err != nil {
		return nextCollectionTime, nil, err
	}

	defer response.Body.Close()

	pageContent, err := io.ReadAll(response.Body)
	if err != nil {
		return nextCollectionTime, nil, err
	}

	var errorSlice []error

	for ind, metricConfig := range collector.configFile.MetricsConfig {
		matchString := collector.info.regexps[ind].FindStringSubmatch(string(pageContent))
		if matchString != nil {
			if metricConfig.DataType == v1.FloatType {
				regVal, err := strconv.ParseFloat(strings.TrimSpace(matchString[1]), 64)
				if err != nil {
					errorSlice = append(errorSlice, err)
				}
				metrics[metricConfig.Name] = []v1.MetricVal{
					{FloatValue: regVal, Timestamp: currentTime},
				}
			} else if metricConfig.DataType == v1.IntType {
				regVal, err := strconv.ParseInt(strings.TrimSpace(matchString[1]), 10, 64)
				if err != nil {
					errorSlice = append(errorSlice, err)
				}
				metrics[metricConfig.Name] = []v1.MetricVal{
					{IntValue: regVal, Timestamp: currentTime},
				}

			} else {
				errorSlice = append(errorSlice, fmt.Errorf("Unexpected value of 'data_type' for metric '%v' in config ", metricConfig.Name))
			}
		} else {
			errorSlice = append(errorSlice, fmt.Errorf("No match found for regexp: %v for metric '%v' in config", metricConfig.Regex, metricConfig.Name))
		}
	}
	return nextCollectionTime, metrics, compileErrors(errorSlice)
}
