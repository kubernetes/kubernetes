// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pubsub

import (
	"reflect"
	"sort"
	"sync"
	"testing"
	"time"

	"golang.org/x/net/context"
)

func TestKeepAlive(t *testing.T) {
	tick := make(chan time.Time)
	deadline := time.Nanosecond * 15
	s := &testService{modDeadlineCalled: make(chan modDeadlineCall)}
	c := &Client{projectID: "projid", s: s}

	checkModDeadlineCall := func(ackIDs []string) {
		got := <-s.modDeadlineCalled
		sort.Strings(got.ackIDs)

		want := modDeadlineCall{
			subName:  "subname",
			deadline: deadline,
			ackIDs:   ackIDs,
		}

		if !reflect.DeepEqual(got, want) {
			t.Errorf("keepalive: got:\n%v\nwant:\n%v", got, want)
		}
	}

	ka := &keepAlive{
		Client:        c,
		Ctx:           context.Background(),
		Sub:           "subname",
		ExtensionTick: tick,
		Deadline:      deadline,
		MaxExtension:  time.Hour,
	}
	ka.Start()

	ka.Add("a")
	ka.Add("b")
	tick <- time.Time{}
	checkModDeadlineCall([]string{"a", "b"})
	ka.Add("c")
	ka.Remove("b")
	tick <- time.Time{}
	checkModDeadlineCall([]string{"a", "c"})
	ka.Remove("a")
	ka.Remove("c")
	ka.Add("d")
	tick <- time.Time{}
	checkModDeadlineCall([]string{"d"})

	ka.Remove("d")
	ka.Stop()
}

// TestKeepAliveStop checks that Stop blocks until all ackIDs have been removed.
func TestKeepAliveStop(t *testing.T) {
	tick := 100 * time.Microsecond
	ticker := time.NewTicker(tick)
	defer ticker.Stop()

	s := &testService{modDeadlineCalled: make(chan modDeadlineCall, 100)}
	c := &Client{projectID: "projid", s: s}

	ka := &keepAlive{
		Client:        c,
		Ctx:           context.Background(),
		ExtensionTick: ticker.C,
		MaxExtension:  time.Hour,
	}
	ka.Start()

	events := make(chan string, 10)

	// Add an ackID so that ka.Stop will not return immediately.
	ka.Add("a")

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		time.Sleep(tick * 10)
		events <- "pre-remove"
		ka.Remove("a")
		time.Sleep(tick * 10)
		events <- "post-second-sleep"
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		events <- "pre-stop"
		ka.Stop()
		events <- "stopped"

	}()

	wg.Wait()
	close(events)
	eventSequence := []string{}
	for e := range events {
		eventSequence = append(eventSequence, e)
	}

	want := []string{"pre-stop", "pre-remove", "stopped", "post-second-sleep"}
	if !reflect.DeepEqual(eventSequence, want) {
		t.Errorf("keepalive eventsequence: got:\n%v\nwant:\n%v", eventSequence, want)
	}
}

// TestMaxExtensionDeadline checks we stop extending after the configured duration.
func TestMaxExtensionDeadline(t *testing.T) {
	ticker := time.NewTicker(100 * time.Microsecond)
	defer ticker.Stop()

	s := &testService{modDeadlineCalled: make(chan modDeadlineCall, 100)}
	c := &Client{projectID: "projid", s: s}

	maxExtension := time.Millisecond
	ka := &keepAlive{
		Client:        c,
		Ctx:           context.Background(),
		ExtensionTick: ticker.C,
		MaxExtension:  maxExtension,
	}
	ka.Start()

	ka.Add("a")
	stopped := make(chan struct{})

	go func() {
		ka.Stop()
		stopped <- struct{}{}
	}()

	select {
	case <-stopped:
	case <-time.After(maxExtension + 2*time.Second):
		t.Fatalf("keepalive failed to stop after maxExtension deadline")
	}
}

func TestKeepAliveStopsImmediatelyForNoAckIDs(t *testing.T) {
	ticker := time.NewTicker(100 * time.Microsecond)
	defer ticker.Stop()

	s := &testService{modDeadlineCalled: make(chan modDeadlineCall, 100)}
	c := &Client{projectID: "projid", s: s}

	maxExtension := time.Millisecond
	ka := &keepAlive{
		Client:        c,
		Ctx:           context.Background(),
		ExtensionTick: ticker.C,
		MaxExtension:  maxExtension,
	}
	ka.Start()

	stopped := make(chan struct{})

	go func() {
		// There are no items in ka, so this should return immediately.
		ka.Stop()
		stopped <- struct{}{}
	}()

	select {
	case <-stopped:
	case <-time.After(maxExtension / 2):
		t.Fatalf("keepalive failed to stop before maxExtension deadline")
	}
}
