/*
Copyright 2015 The Kubernetes Authors.

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

package queue

import (
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/util/clock"
)

func newTestBasicWorkQueue() (*basicWorkQueue, *clock.FakeClock) {
	fakeClock := clock.NewFakeClock(time.Now())
	wq := &basicWorkQueue{
		clock: fakeClock,
		queue: make(map[types.UID]time.Time),
	}
	return wq, fakeClock
}

func compareResults(t *testing.T, expected, actual []types.UID) {
	expectedSet := sets.NewString()
	for _, u := range expected {
		expectedSet.Insert(string(u))
	}
	actualSet := sets.NewString()
	for _, u := range actual {
		actualSet.Insert(string(u))
	}
	if !expectedSet.Equal(actualSet) {
		t.Errorf("Expected %#v, got %#v", expectedSet.List(), actualSet.List())
	}
}

func TestGetWork(t *testing.T) {
	q, clock := newTestBasicWorkQueue()
	q.Enqueue(types.UID("foo1"), -1*time.Minute)
	q.Enqueue(types.UID("foo2"), -1*time.Minute)
	q.Enqueue(types.UID("foo3"), 1*time.Minute)
	q.Enqueue(types.UID("foo4"), 1*time.Minute)
	expected := []types.UID{types.UID("foo1"), types.UID("foo2")}
	compareResults(t, expected, q.GetWork())
	compareResults(t, []types.UID{}, q.GetWork())
	// Dial the time to 1 hour ahead.
	clock.Step(time.Hour)
	expected = []types.UID{types.UID("foo3"), types.UID("foo4")}
	compareResults(t, expected, q.GetWork())
	compareResults(t, []types.UID{}, q.GetWork())
}
