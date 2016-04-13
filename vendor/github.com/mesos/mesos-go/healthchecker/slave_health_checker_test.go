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
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mesos/mesos-go/upid"
	"github.com/stretchr/testify/assert"
)

type thresholdMonitor struct {
	cnt       int32
	threshold int32
}

func newThresholdMonitor(threshold int) *thresholdMonitor {
	return &thresholdMonitor{threshold: int32(threshold)}
}

// incAndTest returns true if the threshold is reached.
func (t *thresholdMonitor) incAndTest() bool {
	if atomic.AddInt32(&t.cnt, 1) >= t.threshold {
		return false
	}
	return true
}

// blockedServer replies only threshold times, after that
// it will block.
type blockedServer struct {
	th       *thresholdMonitor
	ch       chan struct{}
	stopOnce sync.Once
}

func newBlockedServer(threshold int) *blockedServer {
	return &blockedServer{
		th: newThresholdMonitor(threshold),
		ch: make(chan struct{}),
	}
}

func (s *blockedServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if s.th.incAndTest() {
		return
	}
	<-s.ch
}

func (s *blockedServer) stop() {
	s.stopOnce.Do(func() { close(s.ch) })
}

// eofServer will close the connection after it replies for threshold times.
// Thus the health checker will get an EOF error.
type eofServer struct {
	th *thresholdMonitor
}

func newEOFServer(threshold int) *eofServer {
	return &eofServer{newThresholdMonitor(threshold)}
}

func (s *eofServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if s.th.incAndTest() {
		return
	}
	hj := w.(http.Hijacker)
	conn, _, err := hj.Hijack()
	if err != nil {
		panic("Cannot hijack")
	}
	conn.Close()
}

// errorStatusCodeServer will reply error status code (e.g. 503) after
// it replies for threhold time.
type errorStatusCodeServer struct {
	th *thresholdMonitor
}

func newErrorStatusServer(threshold int) *errorStatusCodeServer {
	return &errorStatusCodeServer{newThresholdMonitor(threshold)}
}

func (s *errorStatusCodeServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if s.th.incAndTest() {
		return
	}
	w.WriteHeader(http.StatusServiceUnavailable)
}

// goodServer always returns status ok.
type goodServer bool

func (s *goodServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {}

// partitionedServer returns status ok at some first requests.
// Then it will block for a while, and then reply again.
type partitionedServer struct {
	healthyCnt   int32
	partitionCnt int32
	cnt          int32
	mutex        *sync.Mutex
	cond         *sync.Cond
}

func newPartitionedServer(healthyCnt, partitionCnt int) *partitionedServer {
	mutex := new(sync.Mutex)
	cond := sync.NewCond(mutex)
	return &partitionedServer{
		healthyCnt:   int32(healthyCnt),
		partitionCnt: int32(partitionCnt),
		mutex:        mutex,
		cond:         cond,
	}
}

func (s *partitionedServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	cnt := atomic.AddInt32(&s.cnt, 1)
	if cnt < s.healthyCnt {
		return
	}
	if cnt < s.healthyCnt+s.partitionCnt {
		s.mutex.Lock()
		defer s.mutex.Unlock()
		s.cond.Wait()
		return
	}
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.cond.Broadcast()
}

func TestSlaveHealthCheckerFailedOnBlockedSlave(t *testing.T) {
	s := newBlockedServer(5)
	ts := httptest.NewUnstartedServer(s)
	ts.Start()
	defer ts.Close()
	defer s.stop()

	upid, err := upid.Parse(fmt.Sprintf("slave@%s", ts.Listener.Addr().String()))
	assert.NoError(t, err)

	checker := NewSlaveHealthChecker(upid, 10, time.Millisecond*10, time.Millisecond*10)
	ch := checker.Start()
	defer checker.Stop()

	select {
	case <-time.After(time.Second):
		t.Error("timeout")
	case <-ch:
		assert.True(t, atomic.LoadInt32(&s.th.cnt) > 10)
	}

	// TODO(jdef) hack: this sucks, but there's a data race in httptest's handler when Close()
	// and ServeHTTP() are invoked (WaitGroup DATA RACE). Sleeping here to attempt to avoid that.
	// I think this is supposed to be fixed in go1.6
	time.Sleep(5 * time.Second)
}

func TestSlaveHealthCheckerFailedOnEOFSlave(t *testing.T) {
	s := newEOFServer(5)
	ts := httptest.NewUnstartedServer(s)
	ts.Start()
	defer ts.Close()

	upid, err := upid.Parse(fmt.Sprintf("slave@%s", ts.Listener.Addr().String()))
	assert.NoError(t, err)

	checker := NewSlaveHealthChecker(upid, 10, time.Millisecond*10, time.Millisecond*10)
	ch := checker.Start()
	defer checker.Stop()

	select {
	case <-time.After(time.Second):
		t.Fatal("timeout")
	case <-ch:
		assert.True(t, atomic.LoadInt32(&s.th.cnt) > 10)
	}
}

func TestSlaveHealthCheckerFailedOnErrorStatusSlave(t *testing.T) {
	s := newErrorStatusServer(5)
	ts := httptest.NewUnstartedServer(s)
	ts.Start()
	defer ts.Close()

	upid, err := upid.Parse(fmt.Sprintf("slave@%s", ts.Listener.Addr().String()))
	assert.NoError(t, err)

	checker := NewSlaveHealthChecker(upid, 10, time.Millisecond*10, time.Millisecond*10)
	ch := checker.Start()
	defer checker.Stop()

	select {
	case <-time.After(time.Second):
		t.Fatal("timeout")
	case <-ch:
		assert.True(t, atomic.LoadInt32(&s.th.cnt) > 10)
	}
}

func TestSlaveHealthCheckerSucceed(t *testing.T) {
	t.Skip("skipping known flaky test (fails on busy CI servers, should use a fake clock)")

	s := new(goodServer)
	ts := httptest.NewUnstartedServer(s)
	ts.Start()
	defer ts.Close()

	upid, err := upid.Parse(fmt.Sprintf("slave@%s", ts.Listener.Addr().String()))
	assert.NoError(t, err)

	checker := NewSlaveHealthChecker(upid, 10, time.Millisecond*10, time.Millisecond*10)
	ch := checker.Start()
	defer checker.Stop()

	select {
	case <-time.After(time.Second):
		assert.EqualValues(t, 0, atomic.LoadInt32(&checker.continuousUnhealthyCount))
	case <-ch:
		t.Fatal("Shouldn't get unhealthy notification")
	}
}

func TestSlaveHealthCheckerPartitonedSlave(t *testing.T) {
	t.Skip("skipping known flaky test (fails on busy CI servers, should use a fake clock)")

	s := newPartitionedServer(5, 9)
	ts := httptest.NewUnstartedServer(s)
	ts.Start()
	defer ts.Close()

	t.Log("test server listening on", ts.Listener.Addr())
	upid, err := upid.Parse(fmt.Sprintf("slave@%s", ts.Listener.Addr().String()))
	assert.NoError(t, err)

	checker := NewSlaveHealthChecker(upid, 10, time.Millisecond*10, time.Millisecond*10)
	ch := checker.Start()
	defer func() {
		checker.Stop()
		<-checker.stop
	}()

	select {
	case <-time.After(2 * time.Second):
		actual := atomic.LoadInt32(&checker.continuousUnhealthyCount)
		assert.EqualValues(t, 0, actual, "expected 0 unhealthy counts instead of %d", actual)
	case <-ch:
		t.Fatal("Shouldn't get unhealthy notification")
	}
}
