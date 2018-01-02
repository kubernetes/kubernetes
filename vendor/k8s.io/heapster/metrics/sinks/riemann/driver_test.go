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

package riemann

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	riemann_api "github.com/bigdatadev/goryman"
	"github.com/stretchr/testify/assert"
	"k8s.io/heapster/metrics/core"
)

type eventSendToRiemann struct {
	event string
}

type fakeRiemannClient struct {
	events []eventSendToRiemann
}

type fakeRiemannSink struct {
	core.DataSink
	fakeRiemannClient *fakeRiemannClient
}

func NewFakeRiemannClient() *fakeRiemannClient {
	return &fakeRiemannClient{[]eventSendToRiemann{}}
}

func (client *fakeRiemannClient) Connect() error {
	return nil
}

func (client *fakeRiemannClient) Close() error {
	return nil
}

func (client *fakeRiemannClient) SendEvent(e *riemann_api.Event) error {
	eventsJson, _ := json.Marshal(e)
	client.events = append(client.events, eventSendToRiemann{event: string(eventsJson)})
	return nil
}

// Returns a fake kafka sink.
func NewFakeSink() fakeRiemannSink {
	riemannClient := NewFakeRiemannClient()
	c := riemannConfig{
		host:  "riemann-heapster:5555",
		ttl:   60.0,
		state: "",
		tags:  make([]string, 0),
	}

	return fakeRiemannSink{
		&riemannSink{
			client: riemannClient,
			config: c,
		},
		riemannClient,
	}
}

func TestStoreDataEmptyInput(t *testing.T) {
	fakeSink := NewFakeSink()
	dataBatch := core.DataBatch{}
	fakeSink.ExportData(&dataBatch)
	assert.Equal(t, 0, len(fakeSink.fakeRiemannClient.events))
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

	timeValue := timestamp.Unix()

	fakeSink.ExportData(&data)

	//expect msg string
	assert.Equal(t, 5, len(fakeSink.fakeRiemannClient.events))

	var expectEventsTemplate = [5]string{
		`{"Ttl":60,"Time":%d,"Tags":[],"Host":"","State":"","Service":"/system.slice/-.mount//cpu/limit","Metric":123456,"Description":"","Attributes":{"container_name":"/system.slice/-.mount","namespace_id":"123","pod_id":"aaaa-bbbb-cccc-dddd"}}`,
		`{"Ttl":60,"Time":%d,"Tags":[],"Host":"","State":"","Service":"/system.slice/dbus.service//cpu/usage","Metric":123456,"Description":"","Attributes":{"container_name":"/system.slice/dbus.service","namespace_id":"123","pod_id":"aaaa-bbbb-cccc-dddd"}}`,
		`{"Ttl":60,"Time":%d,"Tags":[],"Host":"","State":"","Service":"test/metric/1","Metric":123456,"Description":"","Attributes":{"namespace_id":"123","pod_id":"aaaa-bbbb-cccc-dddd"}}`,
		`{"Ttl":60,"Time":%d,"Tags":[],"Host":"","State":"","Service":"test/metric/1","Metric":123456,"Description":"","Attributes":{"namespace_id":"","pod_id":"aaaa-bbbb-cccc-dddd"}}`,
		`{"Ttl":60,"Time":%d,"Tags":[],"Host":"","State":"","Service":"removeme","Metric":123456,"Description":"","Attributes":{"namespace_id":"123","pod_id":"aaaa-bbbb-cccc-dddd"}}`,
	}

	eventsString := fmt.Sprintf("%s", fakeSink.fakeRiemannClient.events)
	for _, evtTemplate := range expectEventsTemplate {
		expectEvt := fmt.Sprintf(evtTemplate, timeValue)
		assert.Contains(t, eventsString, expectEvt)
	}
}
