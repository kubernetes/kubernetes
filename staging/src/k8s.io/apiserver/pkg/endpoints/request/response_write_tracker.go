/*
Copyright 2021 The Kubernetes Authors.

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

package request

import (
	"context"
	"sync"
	"time"
)

type responseWriteLatencyTrackerKeyType int

// responseWriteLatencyTrackerKey is the key to associate a response write
// latency tracker with a given request context.
const responseWriteLatencyTrackerKey responseWriteLatencyTrackerKeyType = iota

type ResponseWriteLatencyTracker interface {
	TrackResponseWrite(bytes int, duration time.Duration)
	TrackTransform(items int, duration time.Duration)
	GetMaxPerByteDuration() time.Duration
}

// ResponseWriteLatencyTracker returns a copy of the parent context with
// the given ResponseWriteLatencyTracker instance associated.
func WithResponseWriteLatencyTracker(parent context.Context) context.Context {
	if tracker := ResponseWriteLatencyTrackerFrom(parent); tracker != nil {
		return parent
	}

	return WithValue(parent, responseWriteLatencyTrackerKey, &tracker{})
}

// ResponseWriteLatencyTrackerFrom returns the instance of the ResponseWriteLatencyTracker
// interface associated with the request context.
func ResponseWriteLatencyTrackerFrom(ctx context.Context) ResponseWriteLatencyTracker {
	tracker, _ := ctx.Value(responseWriteLatencyTrackerKey).(ResponseWriteLatencyTracker)
	return tracker
}

type tracker struct {
	lock                 sync.Mutex
	maxByteWriteDuration time.Duration
}

func (t *tracker) TrackResponseWrite(bytes int, duration time.Duration) {
	if bytes <= 0 {
		return
	}

	t.lock.Lock()
	defer t.lock.Unlock()

	this := time.Duration(int64(duration) / int64(bytes))
	if this > t.maxByteWriteDuration {
		t.maxByteWriteDuration = this
	}
}

func (t *tracker) TrackTransform(items int, duration time.Duration) {}

func (t *tracker) GetMaxPerByteDuration() time.Duration {
	t.lock.Lock()
	defer t.lock.Unlock()

	return t.maxByteWriteDuration
}
