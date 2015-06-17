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

package redis

import (
	"encoding/json"
	redis "github.com/garyburd/redigo/redis"
	info "github.com/google/cadvisor/info/v1"
	storage "github.com/google/cadvisor/storage"
	"sync"
	"time"
)

type redisStorage struct {
	conn           redis.Conn
	machineName    string
	redisKey       string
	bufferDuration time.Duration
	lastWrite      time.Time
	lock           sync.Mutex
	readyToFlush   func() bool
}

type detailSpec struct {
	Timestamp      int64                `json:"timestamp"`
	MachineName    string               `json:"machine_name,omitempty"`
	ContainerName  string               `json:"container_Name,omitempty"`
	ContainerStats *info.ContainerStats `json:"container_stats,omitempty"`
}

func (self *redisStorage) defaultReadyToFlush() bool {
	return time.Since(self.lastWrite) >= self.bufferDuration
}

//We must add some defaut params (for example: MachineName,ContainerName...)because containerStats do not include them
func (self *redisStorage) containerStatsAndDefautValues(ref info.ContainerReference, stats *info.ContainerStats) *detailSpec {
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

//Push the data into redis
func (self *redisStorage) AddStats(ref info.ContainerReference, stats *info.ContainerStats) error {
	if stats == nil {
		return nil
	}
	var seriesToFlush []byte
	func() {
		// AddStats will be invoked simultaneously from multiple threads and only one of them will perform a write.
		self.lock.Lock()
		defer self.lock.Unlock()
		// Add some defaut params based on containerStats
		detail := self.containerStatsAndDefautValues(ref, stats)
		//To json
		b, _ := json.Marshal(detail)
		if self.readyToFlush() {
			seriesToFlush = b
			b = nil
			self.lastWrite = time.Now()
		}
	}()
	if len(seriesToFlush) > 0 {
		//We use redis's "LPUSH" to push the data to the redis
		self.conn.Send("LPUSH", self.redisKey, seriesToFlush)
	}
	return nil
}

func (self *redisStorage) Close() error {
	return self.conn.Close()
}

// Create a new redis storage driver.
// machineName: A unique identifier to identify the host that runs the current cAdvisor
// instance is running on.
// redisHost: The host which runs redis.
// redisKey: The key for the Data that stored in the redis
func New(machineName,
	redisKey,
	redisHost string,
	bufferDuration time.Duration,
) (storage.StorageDriver, error) {
	conn, err := redis.Dial("tcp", redisHost)
	if err != nil {
		return nil, err
	}
	ret := &redisStorage{
		conn:           conn,
		machineName:    machineName,
		redisKey:       redisKey,
		bufferDuration: bufferDuration,
		lastWrite:      time.Now(),
	}
	ret.readyToFlush = ret.defaultReadyToFlush
	return ret, nil
}
