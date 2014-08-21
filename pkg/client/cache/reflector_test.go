/*
Copyright 2014 Google Inc. All rights reserved.

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

package cache

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func TestReflector_watchHandler(t *testing.T) {
	s := NewStore()
	g := NewReflector(nil, &api.Pod{}, s)
	fw := watch.NewFake()
	s.Add("foo", &api.Pod{JSONBase: api.JSONBase{ID: "foo"}})
	s.Add("bar", &api.Pod{JSONBase: api.JSONBase{ID: "bar"}})
	go func() {
		fw.Add(&api.Service{JSONBase: api.JSONBase{ID: "rejected"}})
		fw.Delete(&api.Pod{JSONBase: api.JSONBase{ID: "foo"}})
		fw.Modify(&api.Pod{JSONBase: api.JSONBase{ID: "bar", ResourceVersion: 55}})
		fw.Add(&api.Pod{JSONBase: api.JSONBase{ID: "baz", ResourceVersion: 32}})
		fw.Stop()
	}()
	var resumeRV uint64
	g.watchHandler(fw, &resumeRV)

	table := []struct {
		ID     string
		RV     uint64
		exists bool
	}{
		{"foo", 0, false},
		{"rejected", 0, false},
		{"bar", 55, true},
		{"baz", 32, true},
	}
	for _, item := range table {
		obj, exists := s.Get(item.ID)
		if e, a := item.exists, exists; e != a {
			t.Errorf("%v: expected %v, got %v", item.ID, e, a)
		}
		if !exists {
			continue
		}
		if e, a := item.RV, obj.(*api.Pod).ResourceVersion; e != a {
			t.Errorf("%v: expected %v, got %v", item.ID, e, a)
		}
	}

	// RV should stay 1 higher than the last id we see.
	if e, a := uint64(33), resumeRV; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestReflector_Run(t *testing.T) {
	createdFakes := make(chan *watch.FakeWatcher)

	// Expect our starter to get called at the beginning of the watch with 0, and again with 3 when we
	// inject an error at 2.
	expectedRVs := []uint64{0, 3}
	watchStarter := func(rv uint64) (watch.Interface, error) {
		fw := watch.NewFake()
		if e, a := expectedRVs[0], rv; e != a {
			t.Errorf("Expected rv %v, but got %v", e, a)
		}
		expectedRVs = expectedRVs[1:]
		// channel is not buffered because the for loop below needs to block. But
		// we don't want to block here, so report the new fake via a go routine.
		go func() { createdFakes <- fw }()
		return fw, nil
	}
	s := NewFIFO()
	r := NewReflector(watchStarter, &api.Pod{}, s)
	r.period = 0
	r.Run()

	ids := []string{"foo", "bar", "baz", "qux", "zoo"}
	var fw *watch.FakeWatcher
	for i, id := range ids {
		if fw == nil {
			fw = <-createdFakes
		}
		sendingRV := uint64(i + 1)
		fw.Add(&api.Pod{JSONBase: api.JSONBase{ID: id, ResourceVersion: sendingRV}})
		if sendingRV == 2 {
			// Inject a failure.
			fw.Stop()
			fw = nil
		}
	}

	// Verify we received the right ids with the right resource versions.
	for i, id := range ids {
		pod := s.Pop().(*api.Pod)
		if e, a := id, pod.ID; e != a {
			t.Errorf("%v: Expected %v, got %v", i, e, a)
		}
		if e, a := uint64(i+1), pod.ResourceVersion; e != a {
			t.Errorf("%v: Expected %v, got %v", i, e, a)
		}
	}

	if len(expectedRVs) != 0 {
		t.Error("called watchStarter an unexpected number of times")
	}
}
