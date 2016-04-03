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
	"errors"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/upid"
)

const (
	defaultTimeout       = time.Second
	defaultCheckDuration = time.Second
	defaultThreshold     = 5
)

var errCheckerStopped = errors.New("aborted HTTP request because checker was asked to stop")

// SlaveHealthChecker is for checking the slave's health.
type SlaveHealthChecker struct {
	sync.RWMutex
	slaveUPID                *upid.UPID
	tr                       *http.Transport
	client                   *http.Client
	threshold                int32         // marked unhealthy once continuousUnhealthCount is greater than this
	checkDuration            time.Duration // perform the check at this interval
	continuousUnhealthyCount int32         // marked unhealthy when this exceeds threshold
	stop                     chan struct{}
	ch                       chan time.Time
	paused                   bool
}

// NewSlaveHealthChecker creates a slave health checker and return a notification channel.
// Each time the checker thinks the slave is unhealthy, it will send a notification through the channel.
func NewSlaveHealthChecker(slaveUPID *upid.UPID, threshold int, checkDuration time.Duration, timeout time.Duration) *SlaveHealthChecker {
	tr := &http.Transport{}
	checker := &SlaveHealthChecker{
		slaveUPID:     slaveUPID,
		client:        &http.Client{Timeout: timeout, Transport: tr},
		threshold:     int32(threshold),
		checkDuration: checkDuration,
		stop:          make(chan struct{}),
		ch:            make(chan time.Time, 1),
		tr:            tr,
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
		t := time.NewTicker(s.checkDuration)
		defer t.Stop()
		for {
			select {
			case <-t.C:
				select {
				case <-s.stop:
					return
				default:
					// continue
				}
				if paused, slavepid := func() (x bool, y upid.UPID) {
					s.RLock()
					defer s.RUnlock()
					x = s.paused
					if s.slaveUPID != nil {
						y = *s.slaveUPID
					}
					return
				}(); !paused {
					s.doCheck(slavepid)
				}
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

type errHttp struct{ StatusCode int }

func (e *errHttp) Error() string { return fmt.Sprintf("http error code: %d", e.StatusCode) }

func (s *SlaveHealthChecker) doCheck(pid upid.UPID) {
	unhealthy := false
	path := fmt.Sprintf("http://%s:%s/%s/health", pid.Host, pid.Port, pid.ID)
	req, err := http.NewRequest("HEAD", path, nil)
	req.Close = true
	err = s.httpDo(req, func(resp *http.Response, err error) error {
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return &errHttp{resp.StatusCode}
		}
		return nil
	})
	select {
	case <-s.stop:
		return
	default:
	}
	if err != nil {
		log.Errorf("Failed to request the health path: %v", err)
		unhealthy = true
	}
	if unhealthy {
		x := atomic.AddInt32(&s.continuousUnhealthyCount, 1)
		if x >= s.threshold {
			select {
			case s.ch <- time.Now(): // If no one is receiving the channel, then just skip it.
			default:
			}
			atomic.StoreInt32(&s.continuousUnhealthyCount, 0)
		}
		return
	}
	atomic.StoreInt32(&s.continuousUnhealthyCount, 0)
}

func (s *SlaveHealthChecker) httpDo(req *http.Request, f func(*http.Response, error) error) error {
	// Run the HTTP request in a goroutine and pass the response to f.
	c := make(chan error, 1)
	go func() { c <- f(s.client.Do(req)) }()
	select {
	case <-s.stop:
		s.tr.CancelRequest(req)
		<-c // Wait for f to return.
		return errCheckerStopped
	case err := <-c:
		return err
	}
}
