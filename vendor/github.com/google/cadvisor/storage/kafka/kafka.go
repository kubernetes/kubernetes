// Copyright 2016 Google Inc. All Rights Reserved.
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
	"flag"
	"os"
	"strings"
	"time"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/storage"
	"github.com/google/cadvisor/utils/container"

	kafka "github.com/Shopify/sarama"
	"github.com/golang/glog"
)

func init() {
	storage.RegisterStorageDriver("kafka", new)
}

var (
	brokers = flag.String("storage_driver_kafka_broker_list", "localhost:9092", "kafka broker(s) csv")
	topic   = flag.String("storage_driver_kafka_topic", "stats", "kafka topic")
)

type kafkaStorage struct {
	producer    kafka.AsyncProducer
	topic       string
	machineName string
}

type detailSpec struct {
	Timestamp       time.Time            `json:"timestamp"`
	MachineName     string               `json:"machine_name,omitempty"`
	ContainerName   string               `json:"container_Name,omitempty"`
	ContainerID     string               `json:"container_Id,omitempty"`
	ContainerLabels map[string]string    `json:"container_labels,omitempty"`
	ContainerStats  *info.ContainerStats `json:"container_stats,omitempty"`
}

func (driver *kafkaStorage) infoToDetailSpec(ref info.ContainerReference, stats *info.ContainerStats) *detailSpec {
	timestamp := time.Now()
	containerID := ref.Id
	containerLabels := ref.Labels
	containerName := container.GetPreferredName(ref)

	detail := &detailSpec{
		Timestamp:       timestamp,
		MachineName:     driver.machineName,
		ContainerName:   containerName,
		ContainerID:     containerID,
		ContainerLabels: containerLabels,
		ContainerStats:  stats,
	}
	return detail
}

func (driver *kafkaStorage) AddStats(ref info.ContainerReference, stats *info.ContainerStats) error {
	detail := driver.infoToDetailSpec(ref, stats)
	b, err := json.Marshal(detail)

	driver.producer.Input() <- &kafka.ProducerMessage{
		Topic: driver.topic,
		Value: kafka.StringEncoder(b),
	}

	return err
}

func (self *kafkaStorage) Close() error {
	return self.producer.Close()
}

func new() (storage.StorageDriver, error) {
	machineName, err := os.Hostname()
	if err != nil {
		return nil, err
	}
	return newStorage(machineName)
}

func newStorage(machineName string) (storage.StorageDriver, error) {
	config := kafka.NewConfig()
	config.Producer.RequiredAcks = kafka.WaitForAll

	brokerList := strings.Split(*brokers, ",")
	glog.V(4).Infof("Kafka brokers:%q", brokers)

	producer, err := kafka.NewAsyncProducer(brokerList, config)
	if err != nil {
		return nil, err
	}
	ret := &kafkaStorage{
		producer:    producer,
		topic:       *topic,
		machineName: machineName,
	}
	return ret, nil
}
