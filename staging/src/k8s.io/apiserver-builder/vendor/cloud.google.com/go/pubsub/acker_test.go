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

func TestAcker(t *testing.T) {
	tick := make(chan time.Time)
	s := &testService{acknowledgeCalled: make(chan acknowledgeCall)}

	processed := make(chan string, 10)
	acker := &acker{
		s:       s,
		Ctx:     context.Background(),
		Sub:     "subname",
		AckTick: tick,
		Notify:  func(ackID string) { processed <- ackID },
	}
	acker.Start()

	checkAckProcessed := func(ackIDs []string) {
		got := <-s.acknowledgeCalled
		sort.Strings(got.ackIDs)

		want := acknowledgeCall{
			subName: "subname",
			ackIDs:  ackIDs,
		}

		if !reflect.DeepEqual(got, want) {
			t.Errorf("acknowledge: got:\n%v\nwant:\n%v", got, want)
		}
	}

	acker.Ack("a")
	acker.Ack("b")
	tick <- time.Time{}
	checkAckProcessed([]string{"a", "b"})
	acker.Ack("c")
	tick <- time.Time{}
	checkAckProcessed([]string{"c"})
	acker.Stop()

	// all IDS should have been sent to processed.
	close(processed)
	processedIDs := []string{}
	for id := range processed {
		processedIDs = append(processedIDs, id)
	}
	sort.Strings(processedIDs)
	want := []string{"a", "b", "c"}
	if !reflect.DeepEqual(processedIDs, want) {
		t.Errorf("acker processed: got:\n%v\nwant:\n%v", processedIDs, want)
	}
}

func TestAckerFastMode(t *testing.T) {
	tick := make(chan time.Time)
	s := &testService{acknowledgeCalled: make(chan acknowledgeCall)}

	processed := make(chan string, 10)
	acker := &acker{
		s:       s,
		Ctx:     context.Background(),
		Sub:     "subname",
		AckTick: tick,
		Notify:  func(ackID string) { processed <- ackID },
	}
	acker.Start()

	checkAckProcessed := func(ackIDs []string) {
		got := <-s.acknowledgeCalled
		sort.Strings(got.ackIDs)

		want := acknowledgeCall{
			subName: "subname",
			ackIDs:  ackIDs,
		}

		if !reflect.DeepEqual(got, want) {
			t.Errorf("acknowledge: got:\n%v\nwant:\n%v", got, want)
		}
	}
	// No ticks are sent; fast mode doesn't need them.
	acker.Ack("a")
	acker.Ack("b")
	acker.FastMode()
	checkAckProcessed([]string{"a", "b"})
	acker.Ack("c")
	checkAckProcessed([]string{"c"})
	acker.Stop()

	// all IDS should have been sent to processed.
	close(processed)
	processedIDs := []string{}
	for id := range processed {
		processedIDs = append(processedIDs, id)
	}
	sort.Strings(processedIDs)
	want := []string{"a", "b", "c"}
	if !reflect.DeepEqual(processedIDs, want) {
		t.Errorf("acker processed: got:\n%v\nwant:\n%v", processedIDs, want)
	}
}

// TestAckerStop checks that Stop returns immediately.
func TestAckerStop(t *testing.T) {
	tick := make(chan time.Time)
	s := &testService{acknowledgeCalled: make(chan acknowledgeCall, 10)}

	processed := make(chan string)
	acker := &acker{
		s:       s,
		Ctx:     context.Background(),
		Sub:     "subname",
		AckTick: tick,
		Notify:  func(ackID string) { processed <- ackID },
	}

	acker.Start()

	stopped := make(chan struct{})

	acker.Ack("a")

	go func() {
		acker.Stop()
		stopped <- struct{}{}
	}()

	// Stopped should have been written to by the time this sleep completes.
	time.Sleep(time.Millisecond)

	// Receiving from processed should cause Stop to subsequently return,
	// so it should never be possible to read from stopped before
	// processed.
	select {
	case <-stopped:
	case <-processed:
		t.Errorf("acker.Stop processed an ack id before returning")
	case <-time.After(time.Millisecond):
		t.Errorf("acker.Stop never returned")
	}
}

type ackCallResult struct {
	ackIDs []string
	err    error
}

type ackService struct {
	service

	calls []ackCallResult

	t *testing.T // used for error logging.
}

func (as *ackService) acknowledge(ctx context.Context, subName string, ackIDs []string) error {
	if len(as.calls) == 0 {
		as.t.Fatalf("unexpected call to acknowledge: ackIDs: %v", ackIDs)
	}
	call := as.calls[0]
	as.calls = as.calls[1:]

	if got, want := ackIDs, call.ackIDs; !reflect.DeepEqual(got, want) {
		as.t.Errorf("unexpected arguments to acknowledge: got: %v ; want: %v", got, want)
	}
	return call.err
}

// Test implementation returns the first 2 elements as head, and the rest as tail.
func (as *ackService) splitAckIDs(ids []string) ([]string, []string) {
	if len(ids) < 2 {
		return ids, nil
	}
	return ids[:2], ids[2:]
}

func TestAckerSplitsBatches(t *testing.T) {
	type testCase struct {
		calls []ackCallResult
	}
	for _, tc := range []testCase{
		{
			calls: []ackCallResult{
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
			calls: []ackCallResult{
				{
					ackIDs: []string{"a", "b"},
					err:    errors.New("bang"),
				},
				// On error we retry once.
				{
					ackIDs: []string{"a", "b"},
					err:    errors.New("bang"),
				},
				// We give up after failing twice, so we move on to the next set, "c" and "d"
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
		s := &ackService{
			t:     t,
			calls: tc.calls,
		}

		acker := &acker{
			s:      s,
			Ctx:    context.Background(),
			Sub:    "subname",
			Notify: func(string) {},
		}

		acker.ack([]string{"a", "b", "c", "d", "e", "f"})

		if len(s.calls) != 0 {
			t.Errorf("expected ack calls did not occur: %v", s.calls)
		}
	}
}
