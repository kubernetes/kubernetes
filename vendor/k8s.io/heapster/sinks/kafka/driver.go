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
	"encoding/json"
	"fmt"
	"net/url"
	"time"

	"github.com/golang/glog"
	"github.com/optiopay/kafka"
	"github.com/optiopay/kafka/proto"
	"k8s.io/heapster/extpoints"
	sink_api "k8s.io/heapster/sinks/api"
	sinkutil "k8s.io/heapster/sinks/util"
	kube_api "k8s.io/kubernetes/pkg/api"
)

const (
	partition                = 0
	brokerClientID           = "kafka-sink"
	brokerDialTimeout        = 10 * time.Second
	brokerDialRetryLimit     = 1
	brokerDialRetryWait      = 0
	brokerAllowTopicCreation = true
	brokerLeaderRetryLimit   = 1
	brokerLeaderRetryWait    = 0
	timeSeriesTopic          = "heapster-metrics"
	eventsTopic              = "heapster-events"
)

type kafkaSink struct {
	brokerConf      kafka.BrokerConf
	producer        kafka.Producer
	timeSeriesTopic string
	eventsTopic     string
	sinkBrokerHosts []string
	ci              sinkutil.ClientInitializer
}

type KafkaSinkPoint struct {
	MetricsName      string
	MetricsValue     interface{}
	MetricsTimestamp time.Time
	MetricsTags      map[string]string
}

type KafkaSinkEvent struct {
	EventMessage        string
	EventReason         string
	EventTimestamp      time.Time
	EventCount          int
	EventInvolvedObject interface{}
	EventSource         interface{}
}

// START: ExternalSink interface implementations

func (ks *kafkaSink) Register(mds []sink_api.MetricDescriptor) error {
	return nil
}

func (ks *kafkaSink) Unregister(mds []sink_api.MetricDescriptor) error {
	return nil
}

func (ks *kafkaSink) StoreTimeseries(timeseries []sink_api.Timeseries) error {
	if !ks.ci.Done() || timeseries == nil || len(timeseries) <= 0 {
		return nil
	}
	for _, t := range timeseries {
		seriesName := t.Point.Name
		if t.MetricDescriptor.Units.String() != "" {
			seriesName = fmt.Sprintf("%s_%s", seriesName, t.MetricDescriptor.Units.String())
		}
		if t.MetricDescriptor.Type.String() != "" {
			seriesName = fmt.Sprintf("%s_%s", seriesName, t.MetricDescriptor.Type.String())
		}
		sinkPoint := KafkaSinkPoint{
			MetricsName:      seriesName,
			MetricsValue:     t.Point.Value,
			MetricsTimestamp: t.Point.End.UTC(),
			MetricsTags:      make(map[string]string, len(t.Point.Labels)),
		}
		for key, value := range t.Point.Labels {
			if value != "" {
				sinkPoint.MetricsTags[key] = value
			}
		}
		err := ks.produceKafkaMessage(sinkPoint, ks.timeSeriesTopic)
		if err != nil {
			return fmt.Errorf("failed to produce Kafka messages: %s", err)
		}
	}
	return nil
}

func (ks *kafkaSink) StoreEvents(events []kube_api.Event) error {
	if !ks.ci.Done() || events == nil || len(events) <= 0 {
		return nil
	}
	for _, event := range events {
		sinkEvent := KafkaSinkEvent{
			EventMessage:        event.Message,
			EventReason:         event.Reason,
			EventTimestamp:      event.LastTimestamp.UTC(),
			EventCount:          event.Count,
			EventInvolvedObject: event.InvolvedObject,
			EventSource:         event.Source,
		}

		err := ks.produceKafkaMessage(sinkEvent, ks.eventsTopic)
		if err != nil {
			return fmt.Errorf("failed to produce Kafka messages: %s", err)
		}
	}
	return nil
}

// produceKafkaMessage produces messages to kafka
func (ks *kafkaSink) produceKafkaMessage(v interface{}, topic string) error {
	if v == nil {
		return nil
	}
	jsonItems, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("failed to transform the items to json : %s", err)
	}
	message := &proto.Message{Value: []byte(string(jsonItems))}
	_, err = ks.producer.Produce(topic, partition, message)
	if err != nil {
		return fmt.Errorf("failed to produce message to %s:%d: %s", topic, partition, err)
	}
	return nil
}

func (ks *kafkaSink) DebugInfo() string {
	info := fmt.Sprintf("%s\n", ks.Name())
	info += fmt.Sprintf("There are two kafka topics: %s,%s:\n", ks.eventsTopic, ks.timeSeriesTopic)
	info += fmt.Sprintf("Kafka broker list is: %s", ks.sinkBrokerHosts)
	if !ks.ci.Done() {
		info += fmt.Sprintf("Kafka client has not been initialized yet.")
	}
	return info
}

func (ks *kafkaSink) Name() string {
	return "Apache-Kafka Sink"
}

func (ks *kafkaSink) ping() error {
	_, err := kafka.Dial(ks.sinkBrokerHosts, ks.brokerConf)
	return err
}

func (ks *kafkaSink) setupClient() error {
	glog.V(3).Infof("attempting to setup kafka sink")
	broker, err := kafka.Dial(ks.sinkBrokerHosts, ks.brokerConf)
	if err != nil {
		return fmt.Errorf("failed to connect to kafka cluster: %s", err)
	}
	defer broker.Close()
	//create kafka producer
	conf := kafka.NewProducerConf()
	conf.RequiredAcks = proto.RequiredAcksLocal
	sinkProducer := broker.Producer(conf)
	ks.producer = sinkProducer
	glog.V(3).Infof("kafka sink setup successfully")
	return nil
}

func init() {
	extpoints.SinkFactories.Register(NewKafkaSink, "kafka")
}

func NewKafkaSink(uri *url.URL, _ extpoints.HeapsterConf) ([]sink_api.ExternalSink, error) {
	var kafkaSink kafkaSink
	opts, err := url.ParseQuery(uri.RawQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to parser url's query string: %s", err)
	}

	kafkaSink.timeSeriesTopic = timeSeriesTopic
	if len(opts["timeseriestopic"]) > 0 {
		kafkaSink.timeSeriesTopic = opts["timeseriestopic"][0]
	}

	kafkaSink.eventsTopic = eventsTopic
	if len(opts["eventstopic"]) > 0 {
		kafkaSink.eventsTopic = opts["eventstopic"][0]
	}

	if len(opts["brokers"]) < 1 {
		return nil, fmt.Errorf("There is no broker assigned for connecting kafka broker")
	}
	kafkaSink.sinkBrokerHosts = append(kafkaSink.sinkBrokerHosts, opts["brokers"]...)

	glog.V(2).Infof("initializing kafka sink with brokers - %v", kafkaSink.sinkBrokerHosts)
	//connect to kafka cluster
	brokerConf := kafka.NewBrokerConf(brokerClientID)
	brokerConf.DialTimeout = brokerDialTimeout
	brokerConf.DialRetryLimit = brokerDialRetryLimit
	brokerConf.DialRetryWait = brokerDialRetryWait
	brokerConf.LeaderRetryLimit = brokerLeaderRetryLimit
	brokerConf.LeaderRetryWait = brokerLeaderRetryWait
	brokerConf.AllowTopicCreation = true

	// Store broker configuration.
	kafkaSink.brokerConf = brokerConf
	kafkaSink.ci = sinkutil.NewClientInitializer("kafka", kafkaSink.setupClient, kafkaSink.ping, 10*time.Second)
	return []sink_api.ExternalSink{&kafkaSink}, nil
}
