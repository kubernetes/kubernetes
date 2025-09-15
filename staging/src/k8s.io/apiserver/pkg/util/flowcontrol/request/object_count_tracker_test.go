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
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apiserver/pkg/storage"
	testclock "k8s.io/utils/clock/testing"
)

func TestStorageObjectCountTracker(t *testing.T) {
	tests := []struct {
		name          string
		lastUpdated   time.Duration
		skipSetting   bool
		count         int64
		errExpected   error
		countExpected int64
	}{
		{
			name:        "object count not tracked for given resource",
			skipSetting: true,
			errExpected: ObjectCountNotFoundErr,
		},
		{
			name:          "object count is zero",
			count:         0,
			countExpected: 0,
			errExpected:   nil,
		},
		{
			name:          "object count is more than zero",
			count:         799,
			countExpected: 799,
			errExpected:   nil,
		},
		{
			name:          "object count stale",
			count:         799,
			countExpected: 799,
			lastUpdated:   staleTolerationThreshold + time.Millisecond,
			errExpected:   ObjectCountStaleErr,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeClock := &testclock.FakePassiveClock{}
			tracker := &objectCountTracker{
				clock:  fakeClock,
				counts: map[string]*timestampedStats{},
			}

			key := "foo.bar.resource"
			now := time.Now()
			fakeClock.SetTime(now.Add(-test.lastUpdated))
			if !test.skipSetting {
				tracker.Set(key, storage.Stats{ObjectCount: test.count})
			}

			fakeClock.SetTime(now)
			stats, err := tracker.Get(key)
			if test.errExpected != err {
				t.Errorf("Expected error: %v, but got: %v", test.errExpected, err)
			}
			if test.countExpected != stats.ObjectCount {
				t.Errorf("Expected count: %d, but got: %d", test.countExpected, stats.ObjectCount)
			}
			if test.count <= -1 && len(tracker.counts) > 0 {
				t.Errorf("Expected the cache to be empty, but got: %d", len(tracker.counts))
			}
		})
	}
}

func TestStorageObjectCountTrackerWithPrune(t *testing.T) {
	fakeClock := &testclock.FakePassiveClock{}
	tracker := &objectCountTracker{
		clock:  fakeClock,
		counts: map[string]*timestampedStats{},
	}

	now := time.Now()
	fakeClock.SetTime(now.Add(-61 * time.Minute))
	tracker.Set("k1", storage.Stats{ObjectCount: 61})
	fakeClock.SetTime(now.Add(-60 * time.Minute))
	tracker.Set("k2", storage.Stats{ObjectCount: 60})
	// we are going to prune keys that are stale for >= 1h
	// so the above keys are expected to be pruned and the
	// key below should not be pruned.
	mostRecent := now.Add(-59 * time.Minute)
	fakeClock.SetTime(mostRecent)
	tracker.Set("k3", storage.Stats{ObjectCount: 59})
	expected := map[string]*timestampedStats{
		"k3": {
			Stats:         storage.Stats{ObjectCount: 59},
			lastUpdatedAt: mostRecent,
		},
	}

	fakeClock.SetTime(now)
	if err := tracker.prune(time.Hour); err != nil {
		t.Fatalf("Expected no error, but got: %v", err)
	}

	// we expect only one entry in the map, so DeepEqual should work.
	if !reflect.DeepEqual(expected, tracker.counts) {
		t.Errorf("Expected prune to remove stale entries - diff: %s", cmp.Diff(expected, tracker.counts))
	}
}
