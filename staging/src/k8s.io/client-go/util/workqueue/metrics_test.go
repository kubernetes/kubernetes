/*
Copyright 2018 The Kubernetes Authors.

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

package workqueue

import (
	"testing"
	"time"
)

type testMetrics struct {
	added, gotten, finished int64

	updateCalled chan<- struct{}
}

func (m *testMetrics) add(item t)            { m.added++ }
func (m *testMetrics) get(item t)            { m.gotten++ }
func (m *testMetrics) done(item t)           { m.finished++ }
func (m *testMetrics) updateUnfinishedWork() { m.updateCalled <- struct{}{} }

func TestMetrics(t *testing.T) {
	ch := make(chan struct{})
	m := &testMetrics{
		updateCalled: ch,
	}
	q := newQueue("test", m, time.Millisecond)
	<-ch
	q.ShutDown()
	select {
	case <-time.After(time.Second):
		return
	case <-ch:
		t.Errorf("Unexpected update after shutdown was called.")
	}
}
