/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package etcd3

import (
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

// blockLogger is a rate-limited logger that tracks how long a watchChan is blocked.
type blockLogger struct {
	interval      time.Duration
	threshold     time.Duration
	msg           string
	objectType    string
	groupResource schema.GroupResource

	clock clock.Clock

	mu          sync.Mutex
	lastLogTime time.Time
	eventCount  int
	timeWaiting time.Duration
}

func newBlockLogger(interval time.Duration, threshold time.Duration, msg string, objectType string, groupResource schema.GroupResource, clock clock.Clock) *blockLogger {
	l := &blockLogger{
		interval:      interval,
		threshold:     threshold,
		msg:           msg,
		objectType:    objectType,
		groupResource: groupResource,
		clock:         clock,
	}
	l.lastLogTime = l.clock.Now()
	return l
}

func (l *blockLogger) recordWait(waitTime time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.eventCount++
	l.timeWaiting += waitTime

	now := l.clock.Now()
	window := now.Sub(l.lastLogTime)
	if window < l.interval {
		return
	}

	if l.timeWaiting > l.threshold {
		klog.V(3).InfoS(l.msg,
			"objectType", l.objectType,
			"groupResource", l.groupResource,
			"eventCount", l.eventCount,
			"timeWaiting", l.timeWaiting,
			"window", window,
		)
	}
	l.eventCount = 0
	l.timeWaiting = 0
	l.lastLogTime = now
}
