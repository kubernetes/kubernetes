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

package monasca

import (
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/heapster/metrics/core"
)

type monascaSink struct {
	client Client
	sync.RWMutex
	numberOfFailures int
}

// Pushes the specified metric measurement to the Monasca API.
// The Timeseries are transformed to monasca metrics beforehand.
// Timeseries that cannot be translated to monasca metrics are skipped.
func (sink *monascaSink) ExportData(dataBatch *core.DataBatch) {
	sink.Lock()
	defer sink.Unlock()

	metrics := sink.processMetrics(dataBatch)
	code, response, err := sink.client.SendRequest("POST", "/metrics", metrics)
	if err != nil {
		glog.Errorf("%s", err)
		sink.numberOfFailures++
		return
	}
	if code != http.StatusNoContent {
		glog.Error(response)
		sink.numberOfFailures++
	}
}

// a monasca metric definition
type metric struct {
	Name       string            `json:"name"`
	Dimensions map[string]string `json:"dimensions"`
	Timestamp  int64             `json:"timestamp"`
	Value      float64           `json:"value"`
	ValueMeta  map[string]string `json:"value_meta"`
}

func (sink *monascaSink) processMetrics(dataBatch *core.DataBatch) []metric {
	metrics := []metric{}
	for _, metricSet := range dataBatch.MetricSets {
		m := sink.processMetricSet(dataBatch, metricSet)
		metrics = append(metrics, m...)
	}
	return metrics
}

func (sink *monascaSink) processMetricSet(dataBatch *core.DataBatch, metricSet *core.MetricSet) []metric {
	metrics := []metric{}

	// process unlabeled metrics
	for metricName, metricValue := range metricSet.MetricValues {
		m := sink.processMetric(metricSet.Labels, metricName, dataBatch.Timestamp, metricValue.GetValue())
		if nil != m {
			metrics = append(metrics, *m)
		}
	}

	// process labeled metrics
	for _, metric := range metricSet.LabeledMetrics {
		labels := map[string]string{}
		for k, v := range metricSet.Labels {
			labels[k] = v
		}
		for k, v := range metric.Labels {
			labels[k] = v
		}
		m := sink.processMetric(labels, metric.Name, dataBatch.Timestamp, metric.GetValue())
		if nil != m {
			metrics = append(metrics, *m)
		}
	}
	return metrics
}

func (sink *monascaSink) processMetric(labels map[string]string, name string, timestamp time.Time, value interface{}) *metric {
	val, err := sink.convertValue(value)
	if err != nil {
		glog.Warningf("Metric cannot be pushed to monasca. %#v", value)
		return nil
	}
	dims, valueMeta := sink.processLabels(labels)
	m := metric{
		Name:       strings.Replace(name, "/", ".", -1),
		Dimensions: dims,
		Timestamp:  (timestamp.UnixNano() / 1000000),
		Value:      val,
		ValueMeta:  valueMeta,
	}
	return &m
}

// convert the Timeseries value to a monasca value
func (sink *monascaSink) convertValue(val interface{}) (float64, error) {
	switch val.(type) {
	case int:
		return float64(val.(int)), nil
	case int64:
		return float64(val.(int64)), nil
	case bool:
		if val.(bool) {
			return 1.0, nil
		}
		return 0.0, nil
	case float32:
		return float64(val.(float32)), nil
	case float64:
		return val.(float64), nil
	}
	return 0.0, fmt.Errorf("Unsupported monasca metric value type %T", reflect.TypeOf(val))
}

const (
	emptyValue       = "none"
	monascaComponent = "component"
	monascaService   = "service"
	monascaHostname  = "hostname"
)

// preprocesses heapster labels, splitting into monasca dimensions and monasca meta-values
func (sink *monascaSink) processLabels(labels map[string]string) (map[string]string, map[string]string) {
	dims := map[string]string{}
	valueMeta := map[string]string{}

	// labels to dimensions
	dims[monascaComponent] = sink.processDimension(labels[core.LabelPodName.Key])
	dims[monascaHostname] = sink.processDimension(labels[core.LabelHostname.Key])
	dims[core.LabelContainerName.Key] = sink.processDimension(labels[core.LabelContainerName.Key])
	dims[monascaService] = "kubernetes"

	// labels to valueMeta
	for i, v := range labels {
		if i != core.LabelPodName.Key && i != core.LabelHostname.Key &&
			i != core.LabelContainerName.Key && v != "" {
			valueMeta[i] = strings.Replace(v, ",", " ", -1)
		}
	}
	return dims, valueMeta
}

// creates a valid dimension value
func (sink *monascaSink) processDimension(value string) string {
	if value != "" {
		v := strings.Replace(value, "/", ".", -1)
		return strings.Replace(v, ",", " ", -1)
	}
	return emptyValue
}

func (sink *monascaSink) Name() string {
	return "Monasca Sink"
}

func (sink *monascaSink) Stop() {
	// Nothing needs to be done
}

// CreateMonascaSink creates a monasca sink that can consume the Monasca APIs to create metrics.
func CreateMonascaSink(uri *url.URL) (core.DataSink, error) {
	opts := uri.Query()
	config := NewConfig(opts)
	client, err := NewMonascaClient(config)
	if err != nil {
		return nil, err
	}
	monascaSink := monascaSink{client: client}
	glog.Infof("Created Monasca sink. Monasca server running on: %s", client.GetURL().String())
	return &monascaSink, nil
}
