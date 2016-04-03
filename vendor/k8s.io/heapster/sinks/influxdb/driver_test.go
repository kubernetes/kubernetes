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

package influxdb

import (
	"fmt"
	"strings"
	"testing"
	"time"

	influxdb "github.com/influxdb/influxdb/client"
	"github.com/stretchr/testify/assert"
	sink_api "k8s.io/heapster/sinks/api"
	kube_api "k8s.io/kubernetes/pkg/api"
	kube_api_unv "k8s.io/kubernetes/pkg/api/unversioned"
)

type capturedWriteCall struct {
	points []influxdb.Point
}

type fakeInfluxDBClient struct {
	capturedWriteCalls []capturedWriteCall
}

func NewFakeInfluxDBClient() *fakeInfluxDBClient {
	return &fakeInfluxDBClient{[]capturedWriteCall{}}
}

func (sink *fakeInfluxDBClient) Write(batchPoints influxdb.BatchPoints) (*influxdb.Response, error) {
	sink.capturedWriteCalls = append(sink.capturedWriteCalls, capturedWriteCall{batchPoints.Points})
	return nil, nil
}

func (sink *fakeInfluxDBClient) Query(q influxdb.Query) (*influxdb.Response, error) {
	if strings.Contains(q.Command, "CREATE DATABASE") {
		return nil, nil
	}
	return nil, fmt.Errorf("unimplemented")
}

func (sink *fakeInfluxDBClient) Ping() (time.Duration, string, error) {
	return time.Nanosecond, "", fmt.Errorf("unimplemented")
}

type fakeInfluxDBSink struct {
	sink_api.ExternalSink
	fakeClient *fakeInfluxDBClient
}

// Returns a fake influxdb sink.
func NewFakeSink() fakeInfluxDBSink {
	client := NewFakeInfluxDBClient()
	return fakeInfluxDBSink{
		&influxdbSink{
			client: client,
			c: config{
				host:   "hostname",
				dbName: "databaseName",
			},
		},
		client,
	}
}

func TestStoreEventsNilInput(t *testing.T) {
	// Arrange
	fakeSink := NewFakeSink()

	// Act
	err := fakeSink.StoreEvents(nil /*events*/)

	// Assert
	assert.NoError(t, err)
	assert.Equal(t, 0 /* expected */, len(fakeSink.fakeClient.capturedWriteCalls) /* actual */)
}

func TestStoreEventsEmptyInput(t *testing.T) {
	// Arrange
	fakeSink := NewFakeSink()

	// Act
	err := fakeSink.StoreEvents([]kube_api.Event{})

	// Assert
	assert.NoError(t, err)
	assert.Equal(t, 0 /* expected */, len(fakeSink.fakeClient.capturedWriteCalls) /* actual */)
}

func TestStoreEventsSingleEventInput(t *testing.T) {
	// Arrange
	fakeSink := NewFakeSink()
	eventTime := kube_api_unv.Unix(12345, 0)
	eventSourceHostname := "event1HostName"
	eventReason := "event1"
	events := []kube_api.Event{
		{
			Reason:        eventReason,
			LastTimestamp: eventTime,
			Source: kube_api.EventSource{
				Host: eventSourceHostname,
			},
		},
	}

	// Act
	err := fakeSink.StoreEvents(events)

	// Assert
	assert.NoError(t, err)
	assert.Equal(t, 1 /* expected */, len(fakeSink.fakeClient.capturedWriteCalls) /* actual */)
	assert.Equal(t, 1 /* expected */, len(fakeSink.fakeClient.capturedWriteCalls[0].points) /* actual */)
	assert.Equal(t, eventMeasurementName /* expected */, fakeSink.fakeClient.capturedWriteCalls[0].points[0].Measurement /* actual */)
	assert.Equal(t, 2 /* expected */, len(fakeSink.fakeClient.capturedWriteCalls[0].points[0].Tags) /* actual */)
	assert.Equal(t, 1 /* expected */, len(fakeSink.fakeClient.capturedWriteCalls[0].points[0].Fields) /* actual */)
	assert.Equal(t, eventTime.UTC() /* expected */, fakeSink.fakeClient.capturedWriteCalls[0].points[0].Time /* actual */)
	assert.Equal(t, "" /* expected */, fakeSink.fakeClient.capturedWriteCalls[0].points[0].Tags["pod_id"] /* actual */)
	assert.Equal(t, eventSourceHostname /* expected */, fakeSink.fakeClient.capturedWriteCalls[0].points[0].Tags["hostname"] /* actual */)
	assert.Contains(t, fakeSink.fakeClient.capturedWriteCalls[0].points[0].Fields["value"], eventReason)
}

func TestStoreEventsMultipleEventsInput(t *testing.T) {
	// Arrange
	fakeSink := NewFakeSink()
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

	// Act
	err := fakeSink.StoreEvents(events)

	// Assert
	assert.NoError(t, err)
	assert.Equal(t, 1 /* expected */, len(fakeSink.fakeClient.capturedWriteCalls) /* actual */)
	assert.Equal(t, 2 /* expected */, len(fakeSink.fakeClient.capturedWriteCalls[0].points) /* actual */)
	assert.Equal(t, eventMeasurementName /* expected */, fakeSink.fakeClient.capturedWriteCalls[0].points[0].Measurement /* actual */)
	assert.Equal(t, 2 /* expected */, len(fakeSink.fakeClient.capturedWriteCalls[0].points[0].Tags) /* actual */)
	assert.Equal(t, 1 /* expected */, len(fakeSink.fakeClient.capturedWriteCalls[0].points[0].Fields) /* actual */)
	assert.Equal(t, event1Time.UTC() /* expected */, fakeSink.fakeClient.capturedWriteCalls[0].points[0].Time /* actual */)
	assert.Equal(t, "" /* expected */, fakeSink.fakeClient.capturedWriteCalls[0].points[0].Tags["pod_id"] /* actual */)
	assert.Equal(t, event1SourceHostname /* expected */, fakeSink.fakeClient.capturedWriteCalls[0].points[0].Tags["hostname"] /* actual */)
	assert.Contains(t, fakeSink.fakeClient.capturedWriteCalls[0].points[0].Fields["value"], event1Reason)
	assert.Equal(t, event2Time.UTC() /* expected */, fakeSink.fakeClient.capturedWriteCalls[0].points[1].Time /* actual */)
	assert.Equal(t, "" /* expected */, fakeSink.fakeClient.capturedWriteCalls[0].points[1].Tags["pod_id"] /* actual */)
	assert.Equal(t, event2SourceHostname /* expected */, fakeSink.fakeClient.capturedWriteCalls[0].points[1].Tags["hostname"] /* actual */)
	assert.Contains(t, fakeSink.fakeClient.capturedWriteCalls[0].points[1].Fields["value"], event2Reason)

}
