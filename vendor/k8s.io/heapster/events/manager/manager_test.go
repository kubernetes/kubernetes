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

package manager

import (
	"testing"
	"time"

	"k8s.io/heapster/events/core"
	"k8s.io/heapster/events/util"
	kube_api "k8s.io/kubernetes/pkg/api"
)

func TestFlow(t *testing.T) {
	batch := &core.EventBatch{
		Timestamp: time.Now(),
		Events:    []*kube_api.Event{},
	}

	source := util.NewDummySource(batch)
	sink := util.NewDummySink("sink", time.Millisecond)

	manager, _ := NewManager(source, sink, time.Second)
	manager.Start()

	// 4-5 cycles
	time.Sleep(time.Millisecond * 4500)
	manager.Stop()

	if sink.GetExportCount() < 4 || sink.GetExportCount() > 5 {
		t.Fatalf("Wrong number of exports executed: %d", sink.GetExportCount())
	}
}
