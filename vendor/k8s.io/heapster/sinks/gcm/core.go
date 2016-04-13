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

package gcm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/golang/glog"
	"google.golang.org/cloud/compute/metadata"
	"k8s.io/heapster/util"
	"k8s.io/heapster/util/gce"
	"k8s.io/heapster/util/gcstore"

	sink_api "k8s.io/heapster/sinks/api"
)

const GCMAuthScope = "https://www.googleapis.com/auth/monitoring"

type GcmCore struct {
	// Token to use for authentication.
	token gce.AuthTokenProvider

	// TODO(vmarmol): Make this configurable and not only detected.
	// GCE project.
	project string

	// TODO(vmarmol): Also store labels?
	// Map of metrics we currently export.
	exportedMetrics map[string]metricDescriptor

	// The last value we have pushed for every cumulative metric.
	lastValue gcstore.GCStore
}

// GCM request structures for a MetricDescriptor.
type typeDescriptor struct {
	MetricType string `json:"metricType,omitempty"`
	ValueType  string `json:"valueType,omitempty"`
}

type metricDescriptor struct {
	Name           string                     `json:"name,omitempty"`
	Project        string                     `json:"project,omitempty"`
	Description    string                     `json:"description,omitempty"`
	Labels         []sink_api.LabelDescriptor `json:"labels,omitempty"`
	TypeDescriptor typeDescriptor             `json:"typeDescriptor,omitempty"`
}

// Map of metric name to translation function.
var translationFuncs = map[string]func(float64) float64{
	"uptime_rate": func(value float64) float64 {
		// Convert from milliseconds to seconds.
		return value / 1000
	},
	"cpu/usage_rate": func(value float64) float64 {
		// Convert from billionths of a core to millicores.
		return value / 1000000
	},
}

type listMetricsResponse struct {
	Metrics []metricDescriptor `json:"metrics,omitempty"`
}

func (self *GcmCore) defaultUrlPath() string {
	return fmt.Sprintf("https://www.googleapis.com/cloudmonitoring/v2beta2/projects/%s", self.project)
}

func (self *GcmCore) listMetrics() error {
	response := listMetricsResponse{}
	url, err := url.Parse(self.defaultUrlPath() + "/metricDescriptors")
	if err != nil {
		return err
	}
	err = self.sendRequest("GET", url, nil, &response)
	if err != nil {
		glog.Errorf("[GCM] list metrics failed %v", err)
		return err
	}
	apiPrefix := fmt.Sprintf("%s/%s", customApiPrefix, metricDomain)
	for idx, m := range response.Metrics {
		if strings.HasPrefix(m.Name, apiPrefix) {
			self.exportedMetrics[m.Name] = response.Metrics[idx]
		}
	}

	return nil
}

func (self *GcmCore) deleteMetric(metricName string) error {
	url := &url.URL{
		Scheme: "https",
		Host:   "www.googleapis.com",
		Opaque: fmt.Sprintf("%s/metricDescriptors/%s", self.defaultUrlPath(), url.QueryEscape(metricName)),
	}

	err := self.sendRequest("DELETE", url, nil, nil)
	if err != nil {
		glog.V(2).Infof("[GCM] Deleting metric %q failed: %v", metricName, err)
	}
	return err
}

func (self *GcmCore) addMetric(request metricDescriptor) error {
	if existingMetric, found := self.exportedMetrics[request.Name]; found {
		if existingMetric.TypeDescriptor != request.TypeDescriptor {
			if err := self.deleteMetric(request.Name); err != nil {
				return err
			}
		}
	}
	url, err := url.Parse(self.defaultUrlPath() + "/metricDescriptors")
	if err != nil {
		return err
	}
	if err = self.sendRequest("POST", url, request, nil); err == nil {
		glog.V(3).Infof("[GCM] Added metric %q", request.Name)
		// Add metric to exportedMetrics.
		self.exportedMetrics[request.Name] = request
	}
	return err
}

func (self *GcmCore) Register(name, description, metricType, valueType string, labels []sink_api.LabelDescriptor) error {
	// Enforce the most labels that GCM allows.
	if len(labels) > maxNumLabels {
		return fmt.Errorf("metrics cannot have more than %d labels and %q has %d", maxNumLabels, name, len(labels))
	}

	// Ensure all labels are in the correct format.
	for i := range labels {
		labels[i].Key = FullLabelName(labels[i].Key)
	}

	request := metricDescriptor{
		Name:        FullMetricName(name),
		Project:     self.project,
		Description: description,
		Labels:      labels,
		TypeDescriptor: typeDescriptor{
			MetricType: metricType,
			ValueType:  valueType,
		},
	}
	if err := self.addMetric(request); err != nil {
		return err
	}

	return nil
}

func (self *GcmCore) Unregister(name string) error {
	// No-op
	return nil
}

// GCM request structures for writing time-series data.
type timeseriesDescriptor struct {
	Project string            `json:"project,omitempty"`
	Metric  string            `json:"metric,omitempty"`
	Labels  map[string]string `json:"labels,omitempty"`
}

type point struct {
	Start       time.Time `json:"start,omitempty"`
	End         time.Time `json:"end,omitempty"`
	DoubleValue *float64  `json:"doubleValue,omitempty"`
	Int64Value  *int64    `json:"int64Value,omitempty"`
}

type Timeseries struct {
	TimeseriesDescriptor timeseriesDescriptor `json:"timeseriesDesc,omitempty"`
	Point                point                `json:"point,omitempty"`
}

type metricWriteRequest struct {
	Timeseries []Timeseries `json:"timeseries,omitempty"`
}

type lastValueKey struct {
	metricName string
	labels     string
}

type lastValueData struct {
	value     int64
	timestamp time.Time
}

func (self *GcmCore) StoreTimeseries(metrics map[string][]Timeseries) error {
	// Ensure the metrics exist.
	for name := range metrics {
		// TODO: Remove this check if possible.
		if _, ok := self.exportedMetrics[FullMetricName(name)]; !ok {
			return fmt.Errorf("unable to push unknown metric %q", name)
		}
	}

	// Only send one metric of each type per request.
	var lastErr error
	for len(metrics) != 0 {
		var request metricWriteRequest
		for name, values := range metrics {
			// Remove metrics with no more values.
			if len(values) == 0 {
				delete(metrics, name)
				continue
			}

			m := values[0]
			metrics[name] = values[1:]
			request.Timeseries = append(request.Timeseries, m)
		}
		err := self.pushMetrics(&request)
		if err != nil {
			lastErr = err
		}
	}

	return lastErr
}

func (self *GcmCore) GetMetric(metric *sink_api.Point) (*Timeseries, error) {
	// TODO(vmarmol): Validation and cleanup of data.
	// TODO(vmarmol): Handle non-int64 data types. There is an issue with using omitempty since 0 is a valid value for us.
	value, ok := metric.Value.(int64)
	if !ok {
		return nil, fmt.Errorf("non-int64 data not implemented. Seen for metric %q", metric.Name)
	}

	// Use full label names.
	labels := make(map[string]string, len(metric.Labels))
	for key, value := range metric.Labels {
		labels[FullLabelName(key)] = value
	}

	return &Timeseries{
		TimeseriesDescriptor: timeseriesDescriptor{
			Metric: FullMetricName(metric.Name),
			Labels: labels,
		},
		Point: point{
			Start:      metric.Start,
			End:        metric.End,
			Int64Value: &value,
		},
	}, nil
}

func (self *GcmCore) GetEquivalentRateMetric(metric *sink_api.Point) (*Timeseries, error) {
	// TODO(vmarmol): Validation and cleanup of data.
	// TODO(vmarmol): Handle non-int64 data types. There is an issue with using omitempty since 0 is a valid value for us.
	value, ok := metric.Value.(int64)
	if !ok {
		return nil, fmt.Errorf("non-int64 data not implemented. Seen for metric %q", metric.Name)
	}

	// Use full label names.
	labels := make(map[string]string, len(metric.Labels))
	for key, value := range metric.Labels {
		labels[FullLabelName(key)] = value
	}

	rateMetric, exists := gcmRateMetrics[metric.Name]
	if !exists {
		return nil, nil
	}
	key := lastValueKey{
		metricName: FullMetricName(rateMetric.name),
		labels:     util.LabelsToString(labels, ","),
	}
	lastValueRaw := self.lastValue.Get(key)
	self.lastValue.Put(key, lastValueData{
		value:     value,
		timestamp: metric.End,
	})

	// We need two metrics to do a delta, skip first value.
	if lastValueRaw == nil {
		return nil, nil
	}
	lastValue, ok := lastValueRaw.(lastValueData)
	if !ok {
		return nil, nil
	}
	doubleValue := float64(value)
	doubleValue = float64(value-lastValue.value) / float64(metric.End.UnixNano()-lastValue.timestamp.UnixNano()) * float64(time.Second)

	// Translate to a float using the custom translation function.
	if transFunc, ok := translationFuncs[rateMetric.name]; ok {
		doubleValue = transFunc(doubleValue)
	}
	return &Timeseries{
		TimeseriesDescriptor: timeseriesDescriptor{
			Metric: FullMetricName(rateMetric.name),
			Labels: labels,
		},
		Point: point{
			Start:       metric.End,
			End:         metric.End,
			DoubleValue: &doubleValue,
		},
	}, nil
}

func (self *GcmCore) pushMetrics(request *metricWriteRequest) error {
	if len(request.Timeseries) == 0 {
		return nil
	}
	// TODO(vmarmol): Split requests in this case.
	if len(request.Timeseries) > maxTimeseriesPerRequest {
		return fmt.Errorf("unable to write more than %d metrics at once and %d were provided", maxTimeseriesPerRequest, len(request.Timeseries))
	}

	url, err := url.Parse(fmt.Sprintf("%s/timeseries:write", self.defaultUrlPath()))
	if err != nil {
		return err
	}
	const requestAttempts = 3
	for i := 1; i <= requestAttempts; i++ {
		err = self.sendRequest("POST", url, request, nil)
		if err != nil {
			glog.Warningf("[GCM] Push attempt %d failed: %v", i, err)
		} else {
			break
		}
	}
	if err != nil {
		prettyRequest, _ := json.MarshalIndent(request, "", "  ")
		glog.Warningf("[GCM] Pushing %d metrics \n%s\n failed: %v", len(request.Timeseries), string(prettyRequest), err)
	} else {
		glog.V(2).Infof("[GCM] Pushing %d metrics: SUCCESS", len(request.Timeseries))
	}
	return err
}

const (
	// Domain for the metrics.
	metricDomain = "kubernetes.io"

	customApiPrefix = "custom.cloudmonitoring.googleapis.com"

	maxNumLabels = 10

	// The largest number of timeseries we can write to per request.
	maxTimeseriesPerRequest = 200
)

func FullLabelName(name string) string {
	if !strings.Contains(name, "custom.cloudmonitoring.googleapis.com/") && !strings.Contains(name, "compute.googleapis.com") {
		return fmt.Sprintf("custom.cloudmonitoring.googleapis.com/%s/label/%s", metricDomain, name)
	}
	return name
}

func FullMetricName(name string) string {
	if !strings.HasPrefix(name, customApiPrefix) {
		return fmt.Sprintf("%s/%s/%s", customApiPrefix, metricDomain, name)
	}
	return name
}

func (self *GcmCore) sendRequest(method string, url *url.URL, request interface{}, value interface{}) error {
	token, err := self.token.GetToken()
	if err != nil {
		return err
	}
	var rawRequest io.Reader
	if request != nil {
		jsonRequest, err := json.Marshal(request)
		if err != nil {
			return err
		}
		rawRequest = bytes.NewReader(jsonRequest)
	}
	req, err := http.NewRequest(method, url.String(), rawRequest)
	if err != nil {
		return err
	}
	req.URL = url
	req.Header.Add("Content-Type", "application/json")
	req.Header.Add("Authorization", fmt.Sprintf("Bearer %s", token))

	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		return err
	}

	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("request %+v failed with status %q and response: %+v, Body: %q", req, resp.Status, resp, string(body))
	}
	if value != nil {
		err = json.Unmarshal(body, value)
		if err != nil {
			return fmt.Errorf("failed to parse output. Response: %q. Error: %v", string(body), err)
		}
	}

	return nil
}

// Returns a thread-compatible implementation of GCM interactions.
func NewCore() (*GcmCore, error) {
	token, err := gce.NewAuthTokenProvider(GCMAuthScope)
	if err != nil {
		return nil, err
	}

	// Detect project.
	project, err := metadata.ProjectID()
	if err != nil {
		return nil, err
	}

	core := &GcmCore{
		token:           token,
		project:         project,
		exportedMetrics: make(map[string]metricDescriptor),
		lastValue:       gcstore.New(time.Hour),
	}

	// Wait for an initial token.
	_, err = core.token.WaitForToken()
	if err != nil {
		return nil, err
	}

	if err := core.listMetrics(); err != nil {
		return nil, err
	}

	return core, nil
}
