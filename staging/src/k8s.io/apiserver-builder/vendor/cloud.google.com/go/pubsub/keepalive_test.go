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
	"errors"
	"reflect"
	"sort"
	"testing"
	"time"

	"golang.org/x/net/context"
)

func TestKeepAliveExtendsDeadline(t *testing.T) {
	ticker := make(chan time.Time)
	deadline := time.Nanosecond * 15
	s := &testService{modDeadlineCalled: make(chan modDeadlineCall)}

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
		s:             s,
		Ctx:           context.Background(),
		Sub:           "subname",
		ExtensionTick: ticker,
		Deadline:      deadline,
		MaxExtension:  time.Hour,
	}
	ka.Start()

	ka.Add("a")
	ka.Add("b")
	ticker <- time.Time{}
	checkModDeadlineCall([]string{"a", "b"})
	ka.Add("c")
	ka.Remove("b")
	ticker <- time.Time{}
	checkModDeadlineCall([]string{"a", "c"})
	ka.Remove("a")
	ka.Remove("c")
	ka.Add("d")
	ticker <- time.Time{}
	checkModDeadlineCall([]string{"d"})

	ka.Remove("d")
	ka.Stop()
}

func TestKeepAliveStopsWhenNoItem(t *testing.T) {
	ticker := make(chan time.Time)
	stopped := make(chan bool)
	s := &testService{modDeadlineCalled: make(chan modDeadlineCall, 3)}
	ka := &keepAlive{
		s:             s,
		Ctx:           context.Background(),
		ExtensionTick: ticker,
	}

	ka.Start()

	// There should be no call to modifyAckDeadline since there is no item.
	ticker <- time.Time{}

	go func() {
		ka.Stop() // No items; should not block
		if len(s.modDeadlineCalled) > 0 {
			t.Errorf("unexpected extension to non-existent items: %v", <-s.modDeadlineCalled)
		}
		close(stopped)
	}()

	select {
	case <-stopped:
	case <-time.After(time.Second):
		t.Errorf("keepAlive timed out waiting for stop")
	}
}

func TestKeepAliveStopsWhenItemsExpired(t *testing.T) {
	ticker := make(chan time.Time)
	stopped := make(chan bool)
	s := &testService{modDeadlineCalled: make(chan modDeadlineCall, 2)}
	ka := &keepAlive{
		s:             s,
		Ctx:           context.Background(),
		ExtensionTick: ticker,
		MaxExtension:  time.Duration(0), // Should expire items at the first tick.
	}

	ka.Start()
	ka.Add("a")
	ka.Add("b")

	// Wait until the clock advances. Without this loop, this test fails on
	// Windows because the clock doesn't advance at all between ka.Add and the
	// expiration check after the tick is received.
	begin := time.Now()
	for time.Now().Equal(begin) {
		time.Sleep(time.Millisecond)
	}

	// There should be no call to modifyAckDeadline since both items are expired.
	ticker <- time.Time{}

	go func() {
		ka.Stop() // No live items; should not block.
		if len(s.modDeadlineCalled) > 0 {
			t.Errorf("unexpected extension to expired items")
		}
		close(stopped)
	}()

	select {
	case <-stopped:
	case <-time.After(time.Second):
		t.Errorf("timed out waiting for stop")
	}
}

func TestKeepAliveBlocksUntilAllItemsRemoved(t *testing.T) {
	ticker := make(chan time.Time)
	eventc := make(chan string, 3)
	s := &testService{modDeadlineCalled: make(chan modDeadlineCall)}
	ka := &keepAlive{
		s:             s,
		Ctx:           context.Background(),
		ExtensionTick: ticker,
		MaxExtension:  time.Hour, // Should not expire.
	}

	ka.Start()
	ka.Add("a")
	ka.Add("b")

	go func() {
		ticker <- time.Time{}

		// We expect a call since both items should be extended.
		select {
		case args := <-s.modDeadlineCalled:
			sort.Strings(args.ackIDs)
			got := args.ackIDs
			want := []string{"a", "b"}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("mismatching IDs:\ngot  %v\nwant %v", got, want)
			}
		case <-time.After(time.Second):
			t.Errorf("timed out waiting for deadline extend call")
		}

		time.Sleep(10 * time.Millisecond)

		eventc <- "pre-remove-b"
		// Remove one item, Stop should still be waiting.
		ka.Remove("b")

		ticker <- time.Time{}

		// We expect a call since the item is still alive.
		select {
		case args := <-s.modDeadlineCalled:
			got := args.ackIDs
			want := []string{"a"}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("mismatching IDs:\ngot  %v\nwant %v", got, want)
			}
		case <-time.After(time.Second):
			t.Errorf("timed out waiting for deadline extend call")
		}

		time.Sleep(10 * time.Millisecond)

		eventc <- "pre-remove-a"
		// Remove the last item so that Stop can proceed.
		ka.Remove("a")
	}()

	go func() {
		ka.Stop() // Should block all item are removed.
		eventc <- "post-stop"
	}()

	for i, want := range []string{"pre-remove-b", "pre-remove-a", "post-stop"} {
		select {
		case got := <-eventc:
			if got != want {
				t.Errorf("event #%d:\ngot  %v\nwant %v", i, got, want)
			}
		case <-time.After(time.Second):
			t.Errorf("time out waiting for #%d event: want %v", i, want)
		}
	}
}

// extendCallResult contains a list of ackIDs which are expected in an ackID
// extension request, along with the result that should be returned.
type extendCallResult struct {
	ackIDs []string
	err    error
}

// extendService implements modifyAckDeadline using a hard-coded list of extendCallResults.
type extendService struct {
	service

	calls []extendCallResult

	t *testing.T // used for error logging.
}

func (es *extendService) modifyAckDeadline(ctx context.Context, subName string, deadline time.Duration, ackIDs []string) error {
	if len(es.calls) == 0 {
		es.t.Fatalf("unexpected call to modifyAckDeadline: ackIDs: %v", ackIDs)
	}
	call := es.calls[0]
	es.calls = es.calls[1:]

	if got, want := ackIDs, call.ackIDs; !reflect.DeepEqual(got, want) {
		es.t.Errorf("unexpected arguments to modifyAckDeadline: got: %v ; want: %v", got, want)
	}
	return call.err
}

// Test implementation returns the first 2 elements as head, and the rest as tail.
func (es *extendService) splitAckIDs(ids []string) ([]string, []string) {
	if len(ids) < 2 {
		return ids, nil
	}
	return ids[:2], ids[2:]
}

func TestKeepAliveSplitsBatches(t *testing.T) {
	type testCase struct {
		calls []extendCallResult
	}
	for _, tc := range []testCase{
		{
			calls: []extendCallResult{
				{
					ackIDs: []string{"a", "b"},
				},
				{
					ackIDs: []string{"c", "d"},
				},
				{
					ackIDs: []string{"e", "f"},
				},
			},
		},
		{
			calls: []extendCallResult{
				{
					ackIDs: []string{"a", "b"},
					err:    errors.New("bang"),
				},
				// On error we retry once.
				{
					ackIDs: []string{"a", "b"},
					err:    errors.New("bang"),
				},
				// We give up after failing twice, so we move on to the next set, "c" and "d".
				{
					ackIDs: []string{"c", "d"},
					err:    errors.New("bang"),
				},
				// Again, we retry once.
				{
					ackIDs: []string{"c", "d"},
				},
				{
					ackIDs: []string{"e", "f"},
				},
			},
		},
	} {
		s := &extendService{
			t:     t,
			calls: tc.calls,
		}

		ka := &keepAlive{
			s:   s,
			Ctx: context.Background(),
			Sub: "subname",
		}

		ka.extendDeadlines([]string{"a", "b", "c", "d", "e", "f"})

		if len(s.calls) != 0 {
			t.Errorf("expected extend calls did not occur: %v", s.calls)
		}
	}
}
