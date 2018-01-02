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

package sources

import (
	"testing"
	"time"

	"k8s.io/heapster/metrics/util"
)

func TestAllSourcesReplyInTime(t *testing.T) {
	metricsSourceProvider := util.NewDummyMetricsSourceProvider(
		util.NewDummyMetricsSource("s1", time.Second),
		util.NewDummyMetricsSource("s2", time.Second))

	manager, _ := NewSourceManager(metricsSourceProvider, time.Second*3)
	now := time.Now()
	end := now.Truncate(10 * time.Second)
	dataBatch := manager.ScrapeMetrics(end.Add(-10*time.Second), end)

	elapsed := time.Now().Sub(now)
	if elapsed > 3*time.Second {
		t.Fatalf("ScrapeMetrics took too long: %s", elapsed)
	}

	present := make(map[string]bool)
	for key := range dataBatch.MetricSets {
		present[key] = true
	}

	if _, ok := present["s1"]; !ok {
		t.Fatal("s1 not found")
	}

	if _, ok := present["s2"]; !ok {
		t.Fatal("s2 not found")
	}
}

func TestOneSourcesReplyInTime(t *testing.T) {
	metricsSourceProvider := util.NewDummyMetricsSourceProvider(
		util.NewDummyMetricsSource("s1", time.Second),
		util.NewDummyMetricsSource("s2", 30*time.Second))

	manager, _ := NewSourceManager(metricsSourceProvider, time.Second*3)
	now := time.Now()
	end := now.Truncate(10 * time.Second)
	dataBatch := manager.ScrapeMetrics(end.Add(-10*time.Second), end)
	elapsed := time.Now().Sub(now)

	if elapsed > 4*time.Second {
		t.Fatalf("ScrapeMetrics took too long: %s", elapsed)
	}

	if elapsed < 2*time.Second {
		t.Fatalf("ScrapeMetrics took too short: %s", elapsed)
	}

	present := make(map[string]bool)
	for key := range dataBatch.MetricSets {
		present[key] = true
	}

	if _, ok := present["s1"]; !ok {
		t.Fatal("s1 not found")
	}

	if _, ok := present["s2"]; ok {
		t.Fatal("s2 found")
	}
}

func TestNoSourcesReplyInTime(t *testing.T) {
	metricsSourceProvider := util.NewDummyMetricsSourceProvider(
		util.NewDummyMetricsSource("s1", 30*time.Second),
		util.NewDummyMetricsSource("s2", 30*time.Second))

	manager, _ := NewSourceManager(metricsSourceProvider, time.Second*3)
	now := time.Now()
	end := now.Truncate(10 * time.Second)
	dataBatch := manager.ScrapeMetrics(end.Add(-10*time.Second), end)
	elapsed := time.Now().Sub(now)

	if elapsed > 4*time.Second {
		t.Fatalf("ScrapeMetrics took too long: %s", elapsed)
	}

	if elapsed < 2*time.Second {
		t.Fatalf("ScrapeMetrics took too short: %s", elapsed)
	}

	present := make(map[string]bool)
	for key := range dataBatch.MetricSets {
		present[key] = true
	}

	if _, ok := present["s1"]; ok {
		t.Fatal("s1 found")
	}

	if _, ok := present["s2"]; ok {
		t.Fatal("s2 found")
	}
}
