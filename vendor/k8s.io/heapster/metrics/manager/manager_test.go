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

	"k8s.io/heapster/metrics/core"
	"k8s.io/heapster/metrics/util"
)

func TestFlow(t *testing.T) {
	source := util.NewDummyMetricsSource("src", time.Millisecond)
	sink := util.NewDummySink("sink", time.Millisecond)
	processor := util.NewDummyDataProcessor(time.Millisecond)

	manager, _ := NewManager(source, []core.DataProcessor{processor}, sink, time.Second, time.Millisecond, 1)
	manager.Start()

	// 4-5 cycles
	time.Sleep(time.Millisecond * 4500)
	manager.Stop()

	if sink.GetExportCount() < 4 || sink.GetExportCount() > 5 {
		t.Fatalf("Wrong number of exports executed: %d", sink.GetExportCount())
	}
}

func TestThrottling(t *testing.T) {
	source := util.NewDummyMetricsSource("src", time.Millisecond)
	sink := util.NewDummySink("sink", 4*time.Second)
	processor := util.NewDummyDataProcessor(5 * time.Millisecond)

	manager, _ := NewManager(source, []core.DataProcessor{processor}, sink, time.Second, time.Millisecond, 1)
	manager.Start()

	// 4-5 cycles
	time.Sleep(time.Millisecond * 9500)
	manager.Stop()

	if sink.GetExportCount() < 2 || sink.GetExportCount() > 3 {
		t.Fatalf("Wrong number of exports executed: %d", sink.GetExportCount())
	}
}
