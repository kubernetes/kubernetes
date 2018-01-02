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
	"sync"
	"time"

	"github.com/golang/glog"
	"github.com/optiopay/kafka"
	"github.com/optiopay/kafka/proto"
	"k8s.io/heapster/metrics/core"
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
	dataTopic                = "heapster-metrics"
)

type KafkaSinkPoint struct {
	MetricsName      string
	MetricsValue     interface{}
	MetricsTimestamp time.Time
	MetricsTags      map[string]string
}

type kafkaSink struct {
	producer  kafka.Producer
	dataTopic string
	sync.RWMutex
}

func (sink *kafkaSink) ExportData(dataBatch *core.DataBatch) {
	sink.Lock()
	defer sink.Unlock()

	for _, metricSet := range dataBatch.MetricSets {
		for metricName, metricValue := range metricSet.MetricValues {
			point := KafkaSinkPoint{
				MetricsName: metricName,
				MetricsTags: metricSet.Labels,
				MetricsValue: map[string]interface{}{
					"value": metricValue.GetValue(),
				},
				MetricsTimestamp: dataBatch.Timestamp.UTC(),
			}
			sink.produceKafkaMessage(point, sink.dataTopic)
		}
		for _, metric := range metricSet.LabeledMetrics {
			labels := make(map[string]string)
			for k, v := range metricSet.Labels {
				labels[k] = v
			}
			for k, v := range metric.Labels {
				labels[k] = v
			}
			point := KafkaSinkPoint{
				MetricsName: metric.Name,
				MetricsTags: labels,
				MetricsValue: map[string]interface{}{
					"value": metric.GetValue(),
				},
				MetricsTimestamp: dataBatch.Timestamp.UTC(),
			}
			sink.produceKafkaMessage(point, sink.dataTopic)
		}
	}
}

func (sink *kafkaSink) produceKafkaMessage(dataPoint KafkaSinkPoint, topic string) error {
	start := time.Now()
	jsonItems, err := json.Marshal(dataPoint)
	if err != nil {
		return fmt.Errorf("failed to transform the items to json : %s", err)
	}
	message := &proto.Message{Value: []byte(string(jsonItems))}
	_, err = sink.producer.Produce(topic, partition, message)
	if err != nil {
		return fmt.Errorf("failed to produce message to %s:%d: %s", topic, partition, err)
	}
	end := time.Now()
	glog.V(4).Info("Exported %d data to kafka in %s", len([]byte(string(jsonItems))), end.Sub(start))
	return nil
}

func (sink *kafkaSink) Name() string {
	return "Apache Kafka Sink"
}

func (sink *kafkaSink) Stop() {
	// nothing needs to be done.
}

// setupProducer returns a producer of kafka server
func setupProducer(sinkBrokerHosts []string, brokerConf kafka.BrokerConf) (kafka.Producer, error) {
	glog.V(3).Infof("attempting to setup kafka sink")
	broker, err := kafka.Dial(sinkBrokerHosts, brokerConf)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to kafka cluster: %s", err)
	}
	defer broker.Close()

	//create kafka producer
	conf := kafka.NewProducerConf()
	conf.RequiredAcks = proto.RequiredAcksLocal
	sinkProducer := broker.Producer(conf)
	glog.V(3).Infof("kafka sink setup successfully")
	return sinkProducer, nil
}

func NewKafkaSink(uri *url.URL) (core.DataSink, error) {
	opts, err := url.ParseQuery(uri.RawQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to parser url's query string: %s", err)
	}

	var topic string = dataTopic
	if len(opts["timeseriestopic"]) > 0 {
		topic = opts["timeseriestopic"][0]
	}

	var kafkaBrokers []string
	if len(opts["brokers"]) < 1 {
		return nil, fmt.Errorf("There is no broker assigned for connecting kafka")
	}
	kafkaBrokers = append(kafkaBrokers, opts["brokers"]...)
	glog.V(2).Infof("initializing kafka sink with brokers - %v", kafkaBrokers)

	//structure the config of broker
	brokerConf := kafka.NewBrokerConf(brokerClientID)
	brokerConf.DialTimeout = brokerDialTimeout
	brokerConf.DialRetryLimit = brokerDialRetryLimit
	brokerConf.DialRetryWait = brokerDialRetryWait
	brokerConf.LeaderRetryLimit = brokerLeaderRetryLimit
	brokerConf.LeaderRetryWait = brokerLeaderRetryWait
	brokerConf.AllowTopicCreation = true

	// set up producer of kafka server.
	sinkProducer, err := setupProducer(kafkaBrokers, brokerConf)
	if err != nil {
		return nil, fmt.Errorf("Failed to setup Producer: - %v", err)
	}

	return &kafkaSink{
		producer:  sinkProducer,
		dataTopic: topic,
	}, nil
}
