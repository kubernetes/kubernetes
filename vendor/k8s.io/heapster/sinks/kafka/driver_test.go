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

package kafka

import (
	_ "encoding/json"
	"testing"
	"time"

	"fmt"

	"github.com/optiopay/kafka/proto"
	"github.com/stretchr/testify/assert"
	sink_api "k8s.io/heapster/sinks/api"
	sinkutil "k8s.io/heapster/sinks/util"
	kube_api "k8s.io/kubernetes/pkg/api"
	kube_api_unv "k8s.io/kubernetes/pkg/api/unversioned"
)

type msgProducedToKafka struct {
	message string
}

type fakeKafkaProducer struct {
	msgs []msgProducedToKafka
}

type fakeKafkaSink struct {
	sink_api.ExternalSink
	fakeProducer *fakeKafkaProducer
}

func NewFakeKafkaProducer() *fakeKafkaProducer {
	return &fakeKafkaProducer{[]msgProducedToKafka{}}
}

func (producer *fakeKafkaProducer) Produce(topic string, partition int32, messages ...*proto.Message) (int64, error) {
	for _, msg := range messages {
		producer.msgs = append(producer.msgs, msgProducedToKafka{string(msg.Value)})
	}
	return 0, nil
}

// Returns a fake kafka sink.
func NewFakeSink() fakeKafkaSink {
	producer := NewFakeKafkaProducer()
	fakeTimeSeriesTopic := "kafkaTime-test-topic"
	fakeEventsTopic := "kafkaEvent-test-topic"
	fakesinkBrokerHosts := make([]string, 2)
	return fakeKafkaSink{
		&kafkaSink{
			producer:        producer,
			timeSeriesTopic: fakeTimeSeriesTopic,
			eventsTopic:     fakeEventsTopic,
			sinkBrokerHosts: fakesinkBrokerHosts,
			ci:              sinkutil.NewClientInitializer("test", func() error { return nil }, func() error { return nil }, time.Millisecond),
		},
		producer,
	}
}

func TestStoreEventsEmptyInput(t *testing.T) {
	fakeSink := NewFakeSink()
	err := fakeSink.StoreEvents([]kube_api.Event{})
	assert.NoError(t, err)
	assert.Equal(t, 0, len(fakeSink.fakeProducer.msgs))
}

func TestStoreEventsSingleEventInput(t *testing.T) {
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
	//expect msg string
	timeStr, err := eventTime.MarshalJSON()
	assert.NoError(t, err)

	msgString := fmt.Sprintf(`{"EventMessage":"","EventReason":"%s","EventTimestamp":%s,"EventCount":0,"EventInvolvedObject":{},"EventSource":{"host":"%s"}}`, eventReason, string(timeStr), eventSourceHostname)
	err = fakeSink.StoreEvents(events)
	assert.NoError(t, err)

	assert.Equal(t, 1, len(fakeSink.fakeProducer.msgs))
	assert.Equal(t, msgString, fakeSink.fakeProducer.msgs[0].message)
}

func TestStoreEventsMultipleEventsInput(t *testing.T) {
	fakeSink := NewFakeSink()
	eventTime := kube_api_unv.Unix(12345, 0)
	event1SourceHostname := "event1HostName"
	event2SourceHostname := "event2HostName"
	event1Reason := "eventReason1"
	event2Reason := "eventReason2"
	events := []kube_api.Event{
		{
			Reason:        event1Reason,
			LastTimestamp: eventTime,
			Source: kube_api.EventSource{
				Host: event1SourceHostname,
			},
		},
		{
			Reason:        event2Reason,
			LastTimestamp: eventTime,
			Source: kube_api.EventSource{
				Host: event2SourceHostname,
			},
		},
	}
	err := fakeSink.StoreEvents(events)
	assert.NoError(t, err)
	assert.Equal(t, 2, len(fakeSink.fakeProducer.msgs))

	timeStr, err := eventTime.MarshalJSON()
	assert.NoError(t, err)

	msgString1 := fmt.Sprintf(`{"EventMessage":"","EventReason":"%s","EventTimestamp":%s,"EventCount":0,"EventInvolvedObject":{},"EventSource":{"host":"%s"}}`, event1Reason, string(timeStr), event1SourceHostname)
	assert.Equal(t, msgString1, fakeSink.fakeProducer.msgs[0].message)

	msgString2 := fmt.Sprintf(`{"EventMessage":"","EventReason":"%s","EventTimestamp":%s,"EventCount":0,"EventInvolvedObject":{},"EventSource":{"host":"%s"}}`, event2Reason, string(timeStr), event2SourceHostname)
	assert.Equal(t, msgString2, fakeSink.fakeProducer.msgs[1].message)
}

func TestStoreTimeseriesEmptyInput(t *testing.T) {
	fakeSink := NewFakeSink()
	err := fakeSink.StoreTimeseries([]sink_api.Timeseries{})
	assert.NoError(t, err)
	assert.Equal(t, 0, len(fakeSink.fakeProducer.msgs))
}

func TestStoreTimeseriesSingleTimeserieInput(t *testing.T) {
	fakeSink := NewFakeSink()

	smd := sink_api.MetricDescriptor{
		ValueType: sink_api.ValueInt64,
		Type:      sink_api.MetricCumulative,
	}

	l := make(map[string]string)
	l["test"] = "notvisible"
	l[sink_api.LabelHostname.Key] = "localhost"
	l[sink_api.LabelContainerName.Key] = "docker"
	l[sink_api.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"
	timeNow := time.Now()

	p := sink_api.Point{
		Name:   "test/metric/1",
		Labels: l,
		Start:  timeNow,
		End:    timeNow,
		Value:  int64(123456),
	}

	timeseries := []sink_api.Timeseries{
		{
			MetricDescriptor: &smd,
			Point:            &p,
		},
	}

	err := fakeSink.StoreTimeseries(timeseries)
	assert.NoError(t, err)

	assert.Equal(t, 1, len(fakeSink.fakeProducer.msgs))

	timeStr, err := timeNow.UTC().MarshalJSON()
	assert.NoError(t, err)

	msgString := fmt.Sprintf(`{"MetricsName":"test/metric/1_cumulative","MetricsValue":123456,"MetricsTimestamp":%s,"MetricsTags":{"container_name":"docker","hostname":"localhost","pod_id":"aaaa-bbbb-cccc-dddd","test":"notvisible"}}`, timeStr)

	assert.Equal(t, msgString, fakeSink.fakeProducer.msgs[0].message)
}

func TestStoreTimeseriesMultipleTimeseriesInput(t *testing.T) {
	fakeSink := NewFakeSink()

	smd := sink_api.MetricDescriptor{
		ValueType: sink_api.ValueInt64,
		Type:      sink_api.MetricCumulative,
	}

	l := make(map[string]string)
	l["test"] = "notvisible"
	l[sink_api.LabelHostname.Key] = "localhost"
	l[sink_api.LabelContainerName.Key] = "docker"
	l[sink_api.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"
	timeNow := time.Now()

	p1 := sink_api.Point{
		Name:   "test/metric/1",
		Labels: l,
		Start:  timeNow,
		End:    timeNow,
		Value:  int64(123456),
	}

	p2 := sink_api.Point{
		Name:   "test/metric/1",
		Labels: l,
		Start:  timeNow,
		End:    timeNow,
		Value:  int64(654321),
	}

	timeseries := []sink_api.Timeseries{
		{
			MetricDescriptor: &smd,
			Point:            &p1,
		},
		{
			MetricDescriptor: &smd,
			Point:            &p2,
		},
	}

	err := fakeSink.StoreTimeseries(timeseries)
	assert.NoError(t, err)

	assert.Equal(t, 2, len(fakeSink.fakeProducer.msgs))

	timeStr, err := timeNow.UTC().MarshalJSON()
	assert.NoError(t, err)

	msgString1 := fmt.Sprintf(`{"MetricsName":"test/metric/1_cumulative","MetricsValue":123456,"MetricsTimestamp":%s,"MetricsTags":{"container_name":"docker","hostname":"localhost","pod_id":"aaaa-bbbb-cccc-dddd","test":"notvisible"}}`, timeStr)
	assert.Equal(t, msgString1, fakeSink.fakeProducer.msgs[0].message)

	msgString2 := fmt.Sprintf(`{"MetricsName":"test/metric/1_cumulative","MetricsValue":654321,"MetricsTimestamp":%s,"MetricsTags":{"container_name":"docker","hostname":"localhost","pod_id":"aaaa-bbbb-cccc-dddd","test":"notvisible"}}`, timeStr)
	assert.Equal(t, msgString2, fakeSink.fakeProducer.msgs[1].message)
}
