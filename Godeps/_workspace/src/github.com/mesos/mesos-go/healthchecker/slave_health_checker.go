/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package healthchecker

import (
	"fmt"
	"net/http"
	"sync"
	"time"

	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/upid"
)

const (
	defaultTimeout       = time.Second
	defaultCheckDuration = time.Second
	defaultThreshold     = 5
)

// SlaveHealthChecker is for checking the slave's health.
type SlaveHealthChecker struct {
	sync.RWMutex
	slaveUPID                *upid.UPID
	client                   *http.Client
	threshold                int
	checkDuration            time.Duration
	continuousUnhealthyCount int
	stop                     chan struct{}
	ch                       chan time.Time
	paused                   bool
}

// NewSlaveHealthChecker creates a slave health checker and return a notification channel.
// Each time the checker thinks the slave is unhealthy, it will send a notification through the channel.
func NewSlaveHealthChecker(slaveUPID *upid.UPID, threshold int, checkDuration time.Duration, timeout time.Duration) *SlaveHealthChecker {
	checker := &SlaveHealthChecker{
		slaveUPID:     slaveUPID,
		client:        &http.Client{Timeout: timeout},
		threshold:     threshold,
		checkDuration: checkDuration,
		stop:          make(chan struct{}),
		ch:            make(chan time.Time, 1),
	}
	if timeout == 0 {
		checker.client.Timeout = defaultTimeout
	}
	if checkDuration == 0 {
		checker.checkDuration = defaultCheckDuration
	}
	if threshold <= 0 {
		checker.threshold = defaultThreshold
	}
	return checker
}

// Start will start the health checker and returns the notification channel.
func (s *SlaveHealthChecker) Start() <-chan time.Time {
	go func() {
		ticker := time.Tick(s.checkDuration)
		for {
			select {
			case <-ticker:
				s.RLock()
				if !s.paused {
					s.doCheck()
				}
				s.RUnlock()
			case <-s.stop:
				return
			}
		}
	}()
	return s.ch
}

// Pause will pause the slave health checker.
func (s *SlaveHealthChecker) Pause() {
	s.Lock()
	defer s.Unlock()
	s.paused = true
}

// Continue will continue the slave health checker with a new slave upid.
func (s *SlaveHealthChecker) Continue(slaveUPID *upid.UPID) {
	s.Lock()
	defer s.Unlock()
	s.paused = false
	s.slaveUPID = slaveUPID
}

// Stop will stop the slave health checker.
// It should be called only once during the life span of the checker.
func (s *SlaveHealthChecker) Stop() {
	close(s.stop)
}

func (s *SlaveHealthChecker) doCheck() {
	path := fmt.Sprintf("http://%s:%s/%s/health", s.slaveUPID.Host, s.slaveUPID.Port, s.slaveUPID.ID)
	resp, err := s.client.Head(path)
	unhealthy := false
	if err != nil {
		log.Errorf("Failed to request the health path: %v\n", err)
		unhealthy = true
	} else if resp.StatusCode != http.StatusOK {
		log.Errorf("Failed to request the health path: status: %v\n", resp.StatusCode)
		unhealthy = true
	}
	if unhealthy {
		s.continuousUnhealthyCount++
		if s.continuousUnhealthyCount >= s.threshold {
			select {
			case s.ch <- time.Now(): // If no one is receiving the channel, then just skip it.
			default:
			}
			s.continuousUnhealthyCount = 0
		}
		return
	}
	s.continuousUnhealthyCount = 0
	resp.Body.Close()
}
