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

package sinks

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	kube_api "k8s.io/kubernetes/pkg/api"

	"k8s.io/heapster/events/core"
	"k8s.io/heapster/events/util"
)

func doThreeBatches(manager core.EventSink) time.Duration {
	now := time.Now()
	batch := core.EventBatch{
		Timestamp: now,
		Events:    []*kube_api.Event{},
	}

	manager.ExportEvents(&batch)
	manager.ExportEvents(&batch)
	manager.ExportEvents(&batch)

	elapsed := time.Now().Sub(now)
	return elapsed
}

func TestAllExportsInTime(t *testing.T) {
	timeout := 3 * time.Second

	sink1 := util.NewDummySink("s1", time.Second)
	sink2 := util.NewDummySink("s2", time.Second)
	manager, _ := NewEventSinkManager([]core.EventSink{sink1, sink2}, timeout, timeout)

	elapsed := doThreeBatches(manager)
	if elapsed > 2*timeout+2*time.Second {
		t.Fatalf("3xExportEvents took too long: %s", elapsed)
	}

	assert.Equal(t, 3, sink1.GetExportCount())
	assert.Equal(t, 3, sink2.GetExportCount())
}

func TestOneExportInTime(t *testing.T) {
	timeout := 3 * time.Second

	sink1 := util.NewDummySink("s1", time.Second)
	sink2 := util.NewDummySink("s2", 30*time.Second)
	manager, _ := NewEventSinkManager([]core.EventSink{sink1, sink2}, timeout, timeout)

	elapsed := doThreeBatches(manager)
	if elapsed > 2*timeout+2*time.Second {
		t.Fatalf("3xExportEvents took too long: %s", elapsed)
	}
	if elapsed < 2*timeout-1*time.Second {
		t.Fatalf("3xExportEvents took too short: %s", elapsed)
	}

	assert.Equal(t, 3, sink1.GetExportCount())
	assert.Equal(t, 1, sink2.GetExportCount())
}

func TestNoExportInTime(t *testing.T) {
	timeout := 3 * time.Second

	sink1 := util.NewDummySink("s1", 30*time.Second)
	sink2 := util.NewDummySink("s2", 30*time.Second)
	manager, _ := NewEventSinkManager([]core.EventSink{sink1, sink2}, timeout, timeout)

	elapsed := doThreeBatches(manager)
	if elapsed > 2*timeout+2*time.Second {
		t.Fatalf("3xExportEvents took too long: %s", elapsed)
	}
	if elapsed < 2*timeout-1*time.Second {
		t.Fatalf("3xExportEvents took too short: %s", elapsed)
	}

	assert.Equal(t, 1, sink1.GetExportCount())
	assert.Equal(t, 1, sink2.GetExportCount())
}

func TestStop(t *testing.T) {
	timeout := 3 * time.Second

	sink1 := util.NewDummySink("s1", 30*time.Second)
	sink2 := util.NewDummySink("s2", 30*time.Second)
	manager, _ := NewEventSinkManager([]core.EventSink{sink1, sink2}, timeout, timeout)

	now := time.Now()
	manager.Stop()
	elapsed := time.Now().Sub(now)
	if elapsed > time.Second {
		t.Fatalf("stop too long: %s", elapsed)
	}
	time.Sleep(time.Second)

	assert.Equal(t, true, sink1.IsStopped())
	assert.Equal(t, true, sink2.IsStopped())
}
