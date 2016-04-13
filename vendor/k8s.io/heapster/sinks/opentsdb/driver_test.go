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

package opentsdb

import (
	"fmt"
	"net/url"
	"testing"
	"time"

	opentsdb "github.com/bluebreezecf/opentsdb-goclient/client"
	opentsdbcfg "github.com/bluebreezecf/opentsdb-goclient/config"
	"github.com/stretchr/testify/assert"
	"k8s.io/heapster/extpoints"
	sink_api "k8s.io/heapster/sinks/api"
	sink_util "k8s.io/heapster/sinks/util"
	kube_api "k8s.io/kubernetes/pkg/api"
	kube_api_unv "k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/types"
)

var (
	fakeOpenTSDBHost = "192.168.1.8:823"
	fakeNodeIp       = "192.168.1.23"
	fakePodName      = "redis-test"
	fakePodUid       = "redis-test-uid"
	fakeLabel        = map[string]string{
		"name": "redis",
		"io.kubernetes.pod.name": "default/redis-test",
		"pod_id":                 fakePodUid,
		"pod_namespace":          "default",
		"pod_name":               fakePodName,
		"container_name":         "redis",
		"container_base_image":   "kubernetes/redis:v1",
		"namespace_id":           "namespace-test-uid",
		"host_id":                fakeNodeIp,
	}
	errorPingFailed = fmt.Errorf("Failed to connect the target opentsdb.")
	errorPutFailed  = fmt.Errorf("The target opentsdb gets error and failed to store the datapoints.")
)

type fakeOpenTSDBClient struct {
	successfulPing     bool
	successfulPut      bool
	receivedDataPoints []opentsdb.DataPoint
}

func (client *fakeOpenTSDBClient) Ping() error {
	if client.successfulPing {
		return nil
	}
	return errorPingFailed
}

func (client *fakeOpenTSDBClient) Put(datapoints []opentsdb.DataPoint, queryParam string) (*opentsdb.PutResponse, error) {
	if !client.successfulPut {
		return nil, errorPutFailed
	}
	client.receivedDataPoints = append(client.receivedDataPoints, datapoints...)
	putRes := opentsdb.PutResponse{
		StatusCode: 200,
		Failed:     0,
		Success:    int64(len(datapoints)),
	}
	return &putRes, nil
}

type fakeOpenTSDBSink struct {
	sink_api.ExternalSink
	fakeClient *fakeOpenTSDBClient
}

func NewFakeOpenTSDBSink(successfulPing, successfulPut bool) fakeOpenTSDBSink {
	client := &fakeOpenTSDBClient{
		successfulPing: successfulPing,
		successfulPut:  successfulPut,
	}
	cfg := opentsdbcfg.OpenTSDBConfig{OpentsdbHost: fakeOpenTSDBHost}
	return fakeOpenTSDBSink{
		&openTSDBSink{
			client: client,
			config: cfg,
			ci:     sink_util.NewClientInitializer("test", func() error { return nil }, func() error { return nil }, time.Millisecond),
		},
		client,
	}
}

func TestStoreTimeseriesNilInput(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	err := fakeSink.StoreTimeseries(nil)
	assert.NoError(t, err)
	assert.Equal(t, 0, len(fakeSink.fakeClient.receivedDataPoints))
}

func TestStoreTimeseriesEmptyInput(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	err := fakeSink.StoreTimeseries([]sink_api.Timeseries{})
	assert.NoError(t, err)
	assert.Equal(t, 0, len(fakeSink.fakeClient.receivedDataPoints))
}

func TestStoreTimeseriesWithPingFailed(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(false, true)
	seriesList := generateFakeTimeseriesList()
	err := fakeSink.StoreTimeseries(seriesList)
	assert.Equal(t, err, errorPingFailed)
	assert.Equal(t, 0, len(fakeSink.fakeClient.receivedDataPoints))
}

func TestStoreTimeseriesWithPutFailed(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, false)
	seriesList := generateFakeTimeseriesList()
	err := fakeSink.StoreTimeseries(seriesList)
	assert.Equal(t, err, errorPutFailed)
	assert.Equal(t, 0, len(fakeSink.fakeClient.receivedDataPoints))
}

func TestStoreTimeseriesSingleTimeserieInput(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	seriesName := "cpu/limit"
	series := generateFakeTimeseries(seriesName, sink_api.MetricGauge, sink_api.UnitsCount, 1000)
	//Without any labels
	series.Point.Labels = map[string]string{}
	seriesList := []sink_api.Timeseries{series}
	err := fakeSink.StoreTimeseries(seriesList)
	assert.NoError(t, err)
	assert.Equal(t, 1, len(fakeSink.fakeClient.receivedDataPoints))
	assert.Equal(t, "cpu_limit_gauge", fakeSink.fakeClient.receivedDataPoints[0].Metric)
	//tsdbSink.secureTags() add a default tag key and value pair
	assert.Equal(t, 1, len(fakeSink.fakeClient.receivedDataPoints[0].Tags))
	assert.Equal(t, defaultTagValue, fakeSink.fakeClient.receivedDataPoints[0].Tags[defaultTagName])
}

func TestStoreTimeseriesMultipleTimeseriesInput(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	seriesList := generateFakeTimeseriesList()
	err := fakeSink.StoreTimeseries(seriesList)
	assert.NoError(t, err)
	assert.Equal(t, len(seriesList), len(fakeSink.fakeClient.receivedDataPoints))
}

func TestStoreEventsNilInput(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	err := fakeSink.StoreEvents(nil)
	assert.NoError(t, err)
	assert.Equal(t, 0, len(fakeSink.fakeClient.receivedDataPoints))
}

func TestStoreEventsEmptyInput(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	err := fakeSink.StoreEvents([]kube_api.Event{})
	assert.NoError(t, err)
	assert.Equal(t, 0, len(fakeSink.fakeClient.receivedDataPoints))
}

func TestStoreEventsWithPingFailed(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(false, true)
	err := fakeSink.StoreEvents(generateFakeEvents())
	assert.Equal(t, err, errorPingFailed)
	assert.Equal(t, 0, len(fakeSink.fakeClient.receivedDataPoints))
}

func TestStoreEventsWithPutFailed(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, false)
	err := fakeSink.StoreEvents(generateFakeEvents())
	assert.Equal(t, err, errorPutFailed)
	assert.Equal(t, 0, len(fakeSink.fakeClient.receivedDataPoints))
}

func TestStoreEventsSingleEventInput(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	eventTime := kube_api_unv.Unix(12345, 0)
	eventSourceHostname := fakeNodeIp
	eventReason := "created"
	involvedObject := kube_api.ObjectReference{
		Kind:      "Pod",
		Name:      fakePodName,
		UID:       types.UID(fakePodUid),
		Namespace: "default",
	}
	events := []kube_api.Event{
		{
			Reason:        eventReason,
			LastTimestamp: eventTime,
			Source: kube_api.EventSource{
				Host: fakeNodeIp,
			},
			InvolvedObject: involvedObject,
		},
	}

	err := fakeSink.StoreEvents(events)

	assert.NoError(t, err)
	assert.Equal(t, 1, len(fakeSink.fakeClient.receivedDataPoints))
	assert.Equal(t, eventMetricName, fakeSink.fakeClient.receivedDataPoints[0].Metric)
	assert.Equal(t, 4, len(fakeSink.fakeClient.receivedDataPoints[0].Tags))
	assert.Equal(t, eventTime.Time.Unix(), fakeSink.fakeClient.receivedDataPoints[0].Timestamp)
	assert.Equal(t, fakePodUid, fakeSink.fakeClient.receivedDataPoints[0].Tags["pod_id"])
	assert.Equal(t, eventSourceHostname, fakeSink.fakeClient.receivedDataPoints[0].Tags[sink_api.LabelHostname.Key])
	assert.Contains(t, fakeSink.fakeClient.receivedDataPoints[0].Value, eventReason)
}

func TestStoreEventsMultipleEventsInput(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	event1Time := kube_api_unv.Unix(12345, 0)
	event2Time := kube_api_unv.Unix(12366, 0)
	event1SourceHostname := "event1HostName"
	event2SourceHostname := "event2HostName"
	event1Reason := "event1"
	event2Reason := "event2"
	events := []kube_api.Event{
		{
			Reason:        event1Reason,
			LastTimestamp: event1Time,
			Source: kube_api.EventSource{
				Host: event1SourceHostname,
			},
		},
		{
			Reason:        event2Reason,
			LastTimestamp: event2Time,
			Source: kube_api.EventSource{
				Host: event2SourceHostname,
			},
		},
	}
	err := fakeSink.StoreEvents(events)

	assert.NoError(t, err)
	assert.Equal(t, 2, len(fakeSink.fakeClient.receivedDataPoints))
	assert.Equal(t, eventMetricName, fakeSink.fakeClient.receivedDataPoints[0].Metric)
	assert.Equal(t, 2, len(fakeSink.fakeClient.receivedDataPoints[0].Tags))
	assert.Equal(t, event1Time.Time.Unix(), fakeSink.fakeClient.receivedDataPoints[0].Timestamp)
	assert.Equal(t, "", fakeSink.fakeClient.receivedDataPoints[0].Tags["pod_id"])
	assert.Equal(t, event1SourceHostname, fakeSink.fakeClient.receivedDataPoints[0].Tags[sink_api.LabelHostname.Key])
	assert.Contains(t, fakeSink.fakeClient.receivedDataPoints[0].Value, event1Reason)
	assert.Equal(t, eventMetricName, fakeSink.fakeClient.receivedDataPoints[1].Metric)
	assert.Equal(t, 2, len(fakeSink.fakeClient.receivedDataPoints[1].Tags))
	assert.Equal(t, event2Time.Time.Unix(), fakeSink.fakeClient.receivedDataPoints[1].Timestamp)
	assert.Equal(t, "", fakeSink.fakeClient.receivedDataPoints[1].Tags["pod_id"])
	assert.Equal(t, event2SourceHostname, fakeSink.fakeClient.receivedDataPoints[1].Tags[sink_api.LabelHostname.Key])
	assert.Contains(t, fakeSink.fakeClient.receivedDataPoints[1].Value, event2Reason)
}

func TestRegister(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	err := fakeSink.Register([]sink_api.MetricDescriptor{})
	assert.NoError(t, err)
	assert.Nil(t, err)
}

func TestUnregister(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	err := fakeSink.Unregister([]sink_api.MetricDescriptor{})
	assert.NoError(t, err)
	assert.Nil(t, err)
}

func TestName(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	name := fakeSink.Name()
	assert.Equal(t, name, opentsdbSinkName)
}

func TestDebugInfo(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	debugInfo := fakeSink.DebugInfo()
	assert.Contains(t, debugInfo, "Sink Type: OpenTSDB")
	assert.Contains(t, debugInfo, "client: Host "+fakeOpenTSDBHost)
	assert.Contains(t, debugInfo, "Number of write failures:")
}

func TestCreateOpenTSDBSinkWithEmptyInputs(t *testing.T) {
	extSinks, err := CreateOpenTSDBSink(&url.URL{}, extpoints.HeapsterConf{})
	assert.NoError(t, err)
	assert.NotNil(t, extSinks)
	assert.Equal(t, 1, len(extSinks))
	tsdbSink, ok := extSinks[0].(*openTSDBSink)
	assert.Equal(t, true, ok)
	assert.Equal(t, defaultOpentsdbHost, tsdbSink.config.OpentsdbHost)
}

func TestCreateOpenTSDBSinkWithNoEmptyInputs(t *testing.T) {
	fakeOpentsdbHost := "192.168.8.23:4242"
	extSinks, err := CreateOpenTSDBSink(&url.URL{Host: fakeOpentsdbHost}, extpoints.HeapsterConf{})
	assert.NoError(t, err)
	assert.NotNil(t, extSinks)
	assert.Equal(t, 1, len(extSinks))
	tsdbSink, ok := extSinks[0].(*openTSDBSink)
	assert.Equal(t, true, ok)
	assert.Equal(t, fakeOpentsdbHost, tsdbSink.config.OpentsdbHost)
}

func generateFakeEvents() []kube_api.Event {
	event1Time := kube_api_unv.Unix(12345, 0)
	event2Time := kube_api_unv.Unix(12366, 0)
	event1SourceHostname := "event1HostName"
	event2SourceHostname := "event2HostName"
	event1Reason := "event1"
	event2Reason := "event2"
	events := []kube_api.Event{
		{
			Reason:        event1Reason,
			LastTimestamp: event1Time,
			Source: kube_api.EventSource{
				Host: event1SourceHostname,
			},
		},
		{
			Reason:        event2Reason,
			LastTimestamp: event2Time,
			Source: kube_api.EventSource{
				Host: event2SourceHostname,
			},
		},
	}
	return events
}

func generateFakeTimeseriesList() []sink_api.Timeseries {
	timeseriesList := make([]sink_api.Timeseries, 0)

	series := generateFakeTimeseries("cpu/limit", sink_api.MetricGauge, sink_api.UnitsCount, 1000)
	timeseriesList = append(timeseriesList, series)
	series = generateFakeTimeseries("cpu/usage", sink_api.MetricCumulative, sink_api.UnitsNanoseconds, 43363664)
	timeseriesList = append(timeseriesList, series)
	series = generateFakeTimeseries("filesystem/limit", sink_api.MetricGauge, sink_api.UnitsBytes, 42241163264)
	timeseriesList = append(timeseriesList, series)
	series = generateFakeTimeseries("filesystem/usage", sink_api.MetricGauge, sink_api.UnitsBytes, 32768)
	timeseriesList = append(timeseriesList, series)
	series = generateFakeTimeseries("memory/limit", sink_api.MetricGauge, sink_api.UnitsBytes, -1)
	timeseriesList = append(timeseriesList, series)
	series = generateFakeTimeseries("memory/usage", sink_api.MetricGauge, sink_api.UnitsBytes, 487424)
	timeseriesList = append(timeseriesList, series)
	series = generateFakeTimeseries("memory/working_set", sink_api.MetricGauge, sink_api.UnitsBytes, 491520)
	timeseriesList = append(timeseriesList, series)
	series = generateFakeTimeseries("uptime", sink_api.MetricCumulative, sink_api.UnitsMilliseconds, 910823)
	timeseriesList = append(timeseriesList, series)

	return timeseriesList
}

func generateFakeTimeseries(name string, metricType sink_api.MetricType, metricUnits sink_api.MetricUnitsType, value interface{}) sink_api.Timeseries {
	end := time.Now()
	start := end.Add(-10)
	point := sink_api.Point{
		Name:   name,
		Labels: fakeLabel,
		Value:  value,
		Start:  start,
		End:    end,
	}
	metricDesc := sink_api.MetricDescriptor{
		Type:  metricType,
		Units: metricUnits,
	}
	series := sink_api.Timeseries{
		Point:            &point,
		MetricDescriptor: &metricDesc,
	}
	return series
}
