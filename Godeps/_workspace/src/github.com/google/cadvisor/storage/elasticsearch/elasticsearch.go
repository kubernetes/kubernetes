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

package elasticsearch

import (
	"fmt"
	"sync"
	"time"

	info "github.com/google/cadvisor/info/v1"
	storage "github.com/google/cadvisor/storage"
	"gopkg.in/olivere/elastic.v2"
)

type elasticStorage struct {
	client      *elastic.Client
	machineName string
	indexName   string
	typeName    string
	lock        sync.Mutex
}

type detailSpec struct {
	Timestamp      int64                `json:"timestamp"`
	MachineName    string               `json:"machine_name,omitempty"`
	ContainerName  string               `json:"container_Name,omitempty"`
	ContainerStats *info.ContainerStats `json:"container_stats,omitempty"`
}

func (self *elasticStorage) containerStatsAndDefaultValues(
	ref info.ContainerReference, stats *info.ContainerStats) *detailSpec {
	timestamp := stats.Timestamp.UnixNano() / 1E3
	var containerName string
	if len(ref.Aliases) > 0 {
		containerName = ref.Aliases[0]
	} else {
		containerName = ref.Name
	}
	detail := &detailSpec{
		Timestamp:      timestamp,
		MachineName:    self.machineName,
		ContainerName:  containerName,
		ContainerStats: stats,
	}
	return detail
}

func (self *elasticStorage) AddStats(ref info.ContainerReference, stats *info.ContainerStats) error {
	if stats == nil {
		return nil
	}
	func() {
		// AddStats will be invoked simultaneously from multiple threads and only one of them will perform a write.
		self.lock.Lock()
		defer self.lock.Unlock()
		// Add some default params based on ContainerStats
		detail := self.containerStatsAndDefaultValues(ref, stats)
		// Index a cadvisor (using JSON serialization)
		_, err := self.client.Index().
			Index(self.indexName).
			Type(self.typeName).
			BodyJson(detail).
			Do()
		if err != nil {
			// Handle error
			panic(fmt.Errorf("failed to write stats to ElasticSearch- %s", err))
		}
	}()
	return nil
}

func (self *elasticStorage) Close() error {
	self.client = nil
	return nil
}

// machineName: A unique identifier to identify the host that current cAdvisor
// instance is running on.
// ElasticHost: The host which runs ElasticSearch.
func New(machineName,
	indexName,
	typeName,
	elasticHost string,
	enableSniffer bool,
) (storage.StorageDriver, error) {
	// Obtain a client and connect to the default Elasticsearch installation
	// on 127.0.0.1:9200. Of course you can configure your client to connect
	// to other hosts and configure it in various other ways.
	client, err := elastic.NewClient(
		elastic.SetHealthcheck(true),
		elastic.SetSniff(enableSniffer),
		elastic.SetHealthcheckInterval(30*time.Second),
		elastic.SetURL(elasticHost),
	)
	if err != nil {
		// Handle error
		panic(err)
	}

	// Ping the Elasticsearch server to get e.g. the version number
	info, code, err := client.Ping().URL(elasticHost).Do()
	if err != nil {
		// Handle error
		panic(err)
	}
	fmt.Printf("Elasticsearch returned with code %d and version %s", code, info.Version.Number)

	ret := &elasticStorage{
		client:      client,
		machineName: machineName,
		indexName:   indexName,
		typeName:    typeName,
	}
	return ret, nil
}
