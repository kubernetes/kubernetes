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
	"fmt"
	"testing"
	"time"

	"github.com/optiopay/kafka/proto"
	"github.com/stretchr/testify/assert"
	"k8s.io/heapster/metrics/core"
)

type msgProducedToKafka struct {
	message string
}

type fakeKafkaProducer struct {
	msgs []msgProducedToKafka
}

type fakeKafkaSink struct {
	core.DataSink
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
	return fakeKafkaSink{
		&kafkaSink{
			producer:  producer,
			dataTopic: fakeTimeSeriesTopic,
		},
		producer,
	}
}

func TestStoreDataEmptyInput(t *testing.T) {
	fakeSink := NewFakeSink()
	dataBatch := core.DataBatch{}
	fakeSink.ExportData(&dataBatch)
	assert.Equal(t, 0, len(fakeSink.fakeProducer.msgs))
}

func TestStoreMultipleDataInput(t *testing.T) {
	fakeSink := NewFakeSink()
	timestamp := time.Now()

	l := make(map[string]string)
	l["namespace_id"] = "123"
	l["container_name"] = "/system.slice/-.mount"
	l[core.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	l2 := make(map[string]string)
	l2["namespace_id"] = "123"
	l2["container_name"] = "/system.slice/dbus.service"
	l2[core.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	l3 := make(map[string]string)
	l3["namespace_id"] = "123"
	l3[core.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	l4 := make(map[string]string)
	l4["namespace_id"] = ""
	l4[core.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	l5 := make(map[string]string)
	l5["namespace_id"] = "123"
	l5[core.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	metricSet1 := core.MetricSet{
		Labels: l,
		MetricValues: map[string]core.MetricValue{
			"/system.slice/-.mount//cpu/limit": {
				ValueType:  core.ValueInt64,
				MetricType: core.MetricCumulative,
				IntValue:   123456,
			},
		},
	}

	metricSet2 := core.MetricSet{
		Labels: l2,
		MetricValues: map[string]core.MetricValue{
			"/system.slice/dbus.service//cpu/usage": {
				ValueType:  core.ValueInt64,
				MetricType: core.MetricCumulative,
				IntValue:   123456,
			},
		},
	}

	metricSet3 := core.MetricSet{
		Labels: l3,
		MetricValues: map[string]core.MetricValue{
			"test/metric/1": {
				ValueType:  core.ValueInt64,
				MetricType: core.MetricCumulative,
				IntValue:   123456,
			},
		},
	}

	metricSet4 := core.MetricSet{
		Labels: l4,
		MetricValues: map[string]core.MetricValue{
			"test/metric/1": {
				ValueType:  core.ValueInt64,
				MetricType: core.MetricCumulative,
				IntValue:   123456,
			},
		},
	}

	metricSet5 := core.MetricSet{
		Labels: l5,
		MetricValues: map[string]core.MetricValue{
			"removeme": {
				ValueType:  core.ValueInt64,
				MetricType: core.MetricCumulative,
				IntValue:   123456,
			},
		},
	}

	data := core.DataBatch{
		Timestamp: timestamp,
		MetricSets: map[string]*core.MetricSet{
			"pod1": &metricSet1,
			"pod2": &metricSet2,
			"pod3": &metricSet3,
			"pod4": &metricSet4,
			"pod5": &metricSet5,
		},
	}

	timeStr, err := timestamp.UTC().MarshalJSON()
	assert.NoError(t, err)

	fakeSink.ExportData(&data)

	//expect msg string
	assert.Equal(t, 5, len(fakeSink.fakeProducer.msgs))

	var expectMsgTemplate = [5]string{
		`{"MetricsName":"/system.slice/-.mount//cpu/limit","MetricsValue":{"value":123456},"MetricsTimestamp":%s,"MetricsTags":{"container_name":"/system.slice/-.mount","namespace_id":"123","pod_id":"aaaa-bbbb-cccc-dddd"}}`,
		`{"MetricsName":"/system.slice/dbus.service//cpu/usage","MetricsValue":{"value":123456},"MetricsTimestamp":%s,"MetricsTags":{"container_name":"/system.slice/dbus.service","namespace_id":"123","pod_id":"aaaa-bbbb-cccc-dddd"}}`,
		`{"MetricsName":"test/metric/1","MetricsValue":{"value":123456},"MetricsTimestamp":%s,"MetricsTags":{"namespace_id":"123","pod_id":"aaaa-bbbb-cccc-dddd"}}`,
		`{"MetricsName":"test/metric/1","MetricsValue":{"value":123456},"MetricsTimestamp":%s,"MetricsTags":{"namespace_id":"","pod_id":"aaaa-bbbb-cccc-dddd"}}`,
		`{"MetricsName":"removeme","MetricsValue":{"value":123456},"MetricsTimestamp":%s,"MetricsTags":{"namespace_id":"123","pod_id":"aaaa-bbbb-cccc-dddd"}}`,
	}

	msgsString := fmt.Sprintf("%s", fakeSink.fakeProducer.msgs)

	for _, mgsTemplate := range expectMsgTemplate {
		expectMsg := fmt.Sprintf(mgsTemplate, timeStr)
		assert.Contains(t, msgsString, expectMsg)
	}

}
