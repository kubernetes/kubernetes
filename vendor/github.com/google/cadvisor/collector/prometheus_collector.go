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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"time"

	rawmodel "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/info/v1"
)

type PrometheusCollector struct {
	// name of the collector
	name string

	// rate at which metrics are collected
	pollingFrequency time.Duration

	// holds information extracted from the config file for a collector
	configFile Prometheus

	// the metrics to gather (uses a map as a set)
	metricsSet map[string]bool

	// Limit for the number of scaped metrics. If the count is higher,
	// no metrics will be returned.
	metricCountLimit int

	// The Http client to use when connecting to metric endpoints
	httpClient *http.Client
}

// Returns a new collector using the information extracted from the configfile
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

	// TODO : Add checks for validity of config file (eg : Accurate JSON fields)
	return &PrometheusCollector{
		name:             collectorName,
		pollingFrequency: minPollingFrequency,
		configFile:       configInJSON,
		metricsSet:       metricsSet,
		metricCountLimit: metricCountLimit,
		httpClient:       httpClient,
	}, nil
}

// Returns name of the collector
func (collector *PrometheusCollector) Name() string {
	return collector.name
}

func (collector *PrometheusCollector) GetSpec() []v1.MetricSpec {

	response, err := collector.httpClient.Get(collector.configFile.Endpoint.URL)
	if err != nil {
		return nil
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		return nil
	}

	dec := expfmt.NewDecoder(response.Body, expfmt.ResponseFormat(response.Header))

	var specs []v1.MetricSpec

	for {
		d := rawmodel.MetricFamily{}
		if err = dec.Decode(&d); err != nil {
			break
		}
		name := d.GetName()
		if len(name) == 0 {
			continue
		}
		// If metrics to collect is specified, skip any metrics not in the list to collect.
		if _, ok := collector.metricsSet[name]; collector.metricsSet != nil && !ok {
			continue
		}

		spec := v1.MetricSpec{
			Name:   name,
			Type:   metricType(d.GetType()),
			Format: v1.FloatType,
		}
		specs = append(specs, spec)
	}

	if err != nil && err != io.EOF {
		return nil
	}

	return specs
}

// metricType converts Prometheus metric type to cadvisor metric type.
// If there is no mapping then just return the name of the Prometheus metric type.
func metricType(t rawmodel.MetricType) v1.MetricType {
	switch t {
	case rawmodel.MetricType_COUNTER:
		return v1.MetricCumulative
	case rawmodel.MetricType_GAUGE:
		return v1.MetricGauge
	default:
		return v1.MetricType(t.String())
	}
}

type prometheusLabels []*rawmodel.LabelPair

func labelSetToLabelPairs(labels model.Metric) prometheusLabels {
	var promLabels prometheusLabels
	for k, v := range labels {
		name := string(k)
		value := string(v)
		promLabels = append(promLabels, &rawmodel.LabelPair{Name: &name, Value: &value})
	}
	return promLabels
}

func (s prometheusLabels) Len() int      { return len(s) }
func (s prometheusLabels) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

// ByName implements sort.Interface by providing Less and using the Len and
// Swap methods of the embedded PrometheusLabels value.
type byName struct{ prometheusLabels }

func (s byName) Less(i, j int) bool {
	return s.prometheusLabels[i].GetName() < s.prometheusLabels[j].GetName()
}

func prometheusLabelSetToCadvisorLabel(promLabels model.Metric) string {
	labels := labelSetToLabelPairs(promLabels)
	sort.Sort(byName{labels})
	var b bytes.Buffer

	for i, l := range labels {
		if i > 0 {
			b.WriteString("\xff")
		}
		b.WriteString(l.GetName())
		b.WriteString("=")
		b.WriteString(l.GetValue())
	}

	return string(b.Bytes())
}

// Returns collected metrics and the next collection time of the collector
func (collector *PrometheusCollector) Collect(metrics map[string][]v1.MetricVal) (time.Time, map[string][]v1.MetricVal, error) {
	currentTime := time.Now()
	nextCollectionTime := currentTime.Add(time.Duration(collector.pollingFrequency))

	uri := collector.configFile.Endpoint.URL
	response, err := collector.httpClient.Get(uri)
	if err != nil {
		return nextCollectionTime, nil, err
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		return nextCollectionTime, nil, fmt.Errorf("server returned HTTP status %s", response.Status)
	}

	sdec := expfmt.SampleDecoder{
		Dec: expfmt.NewDecoder(response.Body, expfmt.ResponseFormat(response.Header)),
		Opts: &expfmt.DecodeOptions{
			Timestamp: model.TimeFromUnixNano(currentTime.UnixNano()),
		},
	}

	var (
		// 50 is chosen as a reasonable guesstimate at a number of metrics we can
		// expect from virtually any endpoint to try to save allocations.
		decSamples = make(model.Vector, 0, 50)
		newMetrics = make(map[string][]v1.MetricVal)
	)
	for {
		if err = sdec.Decode(&decSamples); err != nil {
			break
		}

		for _, sample := range decSamples {
			metName := string(sample.Metric[model.MetricNameLabel])
			if len(metName) == 0 {
				continue
			}
			// If metrics to collect is specified, skip any metrics not in the list to collect.
			if _, ok := collector.metricsSet[metName]; collector.metricsSet != nil && !ok {
				continue
			}
			// TODO Handle multiple labels nicer. Prometheus metrics can have multiple
			// labels, cadvisor only accepts a single string for the metric label.
			label := prometheusLabelSetToCadvisorLabel(sample.Metric)

			metric := v1.MetricVal{
				FloatValue: float64(sample.Value),
				Timestamp:  sample.Timestamp.Time(),
				Label:      label,
			}
			newMetrics[metName] = append(newMetrics[metName], metric)
			if len(newMetrics) > collector.metricCountLimit {
				return nextCollectionTime, nil, fmt.Errorf("too many metrics to collect")
			}
		}
		decSamples = decSamples[:0]
	}

	if err != nil && err != io.EOF {
		return nextCollectionTime, nil, err
	}

	for key, val := range newMetrics {
		metrics[key] = append(metrics[key], val...)
	}

	return nextCollectionTime, metrics, nil
}
