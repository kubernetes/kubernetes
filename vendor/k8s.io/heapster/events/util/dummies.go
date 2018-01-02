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

package util

import (
	"sync"
	"time"

	"k8s.io/heapster/events/core"
)

type DummySink struct {
	name        string
	mutex       sync.Mutex
	exportCount int
	stopped     bool
	latency     time.Duration
}

func (this *DummySink) Name() string {
	return this.name
}
func (this *DummySink) ExportEvents(*core.EventBatch) {
	this.mutex.Lock()
	this.exportCount++
	this.mutex.Unlock()

	time.Sleep(this.latency)
}

func (this *DummySink) Stop() {
	this.mutex.Lock()
	this.stopped = true
	this.mutex.Unlock()

	time.Sleep(this.latency)
}

func (this *DummySink) IsStopped() bool {
	this.mutex.Lock()
	defer this.mutex.Unlock()
	return this.stopped
}

func (this *DummySink) GetExportCount() int {
	this.mutex.Lock()
	defer this.mutex.Unlock()
	return this.exportCount
}

func NewDummySink(name string, latency time.Duration) *DummySink {
	return &DummySink{
		name:        name,
		latency:     latency,
		exportCount: 0,
		stopped:     false,
	}
}

type DummyEventSource struct {
	eventBatch *core.EventBatch
}

func (this *DummyEventSource) GetNewEvents() *core.EventBatch {
	return this.eventBatch
}

func NewDummySource(eventBatch *core.EventBatch) *DummyEventSource {
	return &DummyEventSource{
		eventBatch: eventBatch,
	}
}
