// Copyright 2026 The etcd Authors
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

package clientv3

import (
	"sync"
	"time"
)

type blockLogger struct {
	interval  time.Duration
	threshold time.Duration
	now       func() time.Time
	logFunc   func(eventCount int, timeWaiting time.Duration, window time.Duration)

	mu          sync.Mutex
	lastLogTime time.Time
	eventCount  int
	timeWaiting time.Duration
}

func newBlockLogger(interval time.Duration, threshold time.Duration, now func() time.Time, logFunc func(eventCount int, timeWaiting time.Duration, window time.Duration)) *blockLogger {
	if now == nil {
		now = time.Now
	}
	l := &blockLogger{
		interval:  interval,
		threshold: threshold,
		now:       now,
		logFunc:   logFunc,
	}
	l.lastLogTime = l.now()
	return l
}

func (l *blockLogger) recordWait(waitTime time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.eventCount++
	l.timeWaiting += waitTime

	now := l.now()
	window := now.Sub(l.lastLogTime)
	if window < l.interval {
		return
	}

	if l.timeWaiting > l.threshold && l.logFunc != nil {
		l.logFunc(l.eventCount, l.timeWaiting, window)
	}
	l.eventCount = 0
	l.timeWaiting = 0
	l.lastLogTime = now
}
