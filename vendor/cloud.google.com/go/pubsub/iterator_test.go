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
	"fmt"
	"reflect"
	"testing"
	"time"

	"golang.org/x/net/context"
)

func TestReturnsDoneOnStop(t *testing.T) {
	type testCase struct {
		abort func(*Iterator, context.CancelFunc)
		want  error
	}

	for _, tc := range []testCase{
		{
			abort: func(it *Iterator, cancel context.CancelFunc) {
				it.Stop()
			},
			want: Done,
		},
		{
			abort: func(it *Iterator, cancel context.CancelFunc) {
				cancel()
			},
			want: context.Canceled,
		},
		{
			abort: func(it *Iterator, cancel context.CancelFunc) {
				it.Stop()
				cancel()
			},
			want: Done,
		},
		{
			abort: func(it *Iterator, cancel context.CancelFunc) {
				cancel()
				it.Stop()
			},
			want: Done,
		},
	} {
		s := &blockingFetch{}
		ctx, cancel := context.WithCancel(context.Background())
		it := newIterator(ctx, s, "subname", &pullOptions{ackDeadline: time.Second * 10, maxExtension: time.Hour})
		defer it.Stop()
		tc.abort(it, cancel)

		_, err := it.Next()
		if err != tc.want {
			t.Errorf("iterator Next error after abort: got:\n%v\nwant:\n%v", err, tc.want)
		}
	}
}

// blockingFetch implements message fetching by not returning until its context is cancelled.
type blockingFetch struct {
	service
}

func (s *blockingFetch) fetchMessages(ctx context.Context, subName string, maxMessages int64) ([]*Message, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

// justInTimeFetch simulates the situation where the iterator is aborted just after the fetch RPC
// succeeds, so the rest of puller.Next will continue to execute and return sucessfully.
type justInTimeFetch struct {
	service
}

func (s *justInTimeFetch) fetchMessages(ctx context.Context, subName string, maxMessages int64) ([]*Message, error) {
	<-ctx.Done()
	// The context was cancelled, but let's pretend that this happend just after our RPC returned.

	var result []*Message
	for i := 0; i < int(maxMessages); i++ {
		val := fmt.Sprintf("msg%v", i)
		result = append(result, &Message{Data: []byte(val), ackID: val})
	}
	return result, nil
}

func (s *justInTimeFetch) splitAckIDs(ids []string) ([]string, []string) {
	return nil, nil
}

func (s *justInTimeFetch) modifyAckDeadline(ctx context.Context, subName string, deadline time.Duration, ackIDs []string) error {
	return nil
}

func TestAfterAbortReturnsNoMoreThanOneMessage(t *testing.T) {
	// Each test case is excercised by making two concurrent blocking calls on an
	// Iterator, and then aborting the iterator.
	// The result should be one call to Next returning a message, and the other returning an error.
	type testCase struct {
		abort func(*Iterator, context.CancelFunc)
		// want is the error that should be returned from one Next invocation.
		want error
	}
	for n := 1; n < 3; n++ {
		for _, tc := range []testCase{
			{
				abort: func(it *Iterator, cancel context.CancelFunc) {
					it.Stop()
				},
				want: Done,
			},
			{
				abort: func(it *Iterator, cancel context.CancelFunc) {
					cancel()
				},
				want: context.Canceled,
			},
			{
				abort: func(it *Iterator, cancel context.CancelFunc) {
					it.Stop()
					cancel()
				},
				want: Done,
			},
			{
				abort: func(it *Iterator, cancel context.CancelFunc) {
					cancel()
					it.Stop()
				},
				want: Done,
			},
		} {
			s := &justInTimeFetch{}
			ctx, cancel := context.WithCancel(context.Background())

			// if maxPrefetch == 1, there will be no messages in the puller buffer when Next is invoked the second time.
			// if maxPrefetch == 2, there will be 1 message in the puller buffer when Next is invoked the second time.
			po := &pullOptions{
				ackDeadline:  time.Second * 10,
				maxExtension: time.Hour,
				maxPrefetch:  n,
			}
			it := newIterator(ctx, s, "subname", po)
			defer it.Stop()

			type result struct {
				m   *Message
				err error
			}
			results := make(chan *result, 2)

			for i := 0; i < 2; i++ {
				go func() {
					m, err := it.Next()
					results <- &result{m, err}
					if err == nil {
						m.Done(false)
					}
				}()
			}
			// Wait for goroutines to block on it.Next().
			time.Sleep(time.Millisecond)
			tc.abort(it, cancel)

			result1 := <-results
			result2 := <-results

			// There should be one error result, and one non-error result.
			// Make result1 be the non-error result.
			if result1.err != nil {
				result1, result2 = result2, result1
			}

			if string(result1.m.Data) != "msg0" {
				t.Errorf("After abort, got message: %v, want %v", result1.m.Data, "msg0")
			}
			if result1.err != nil {
				t.Errorf("After abort, got : %v, want nil", result1.err)
			}
			if result2.m != nil {
				t.Errorf("After abort, got message: %v, want nil", result2.m)
			}
			if result2.err != tc.want {
				t.Errorf("After abort, got err: %v, want %v", result2.err, tc.want)
			}
		}
	}
}

func TestMultipleStopCallsBlockUntilMessageDone(t *testing.T) {
	s := &fetcherService{
		results: []fetchResult{
			{
				msgs: []*Message{{ackID: "a"}, {ackID: "b"}},
			},
		},
	}

	ctx := context.Background()
	it := newIterator(ctx, s, "subname", &pullOptions{ackDeadline: time.Second * 10, maxExtension: 0})

	m, err := it.Next()
	if err != nil {
		t.Errorf("error calling Next: %v", err)
	}

	events := make(chan string, 3)
	go func() {
		it.Stop()
		events <- "stopped"
	}()
	go func() {
		it.Stop()
		events <- "stopped"
	}()

	time.Sleep(10 * time.Millisecond)
	events <- "nacked"
	m.Done(false)

	if got, want := []string{<-events, <-events, <-events}, []string{"nacked", "stopped", "stopped"}; !reflect.DeepEqual(got, want) {
		t.Errorf("stopping iterator, got: %v ; want: %v", got, want)
	}

	// The iterator is stopped, so should not return another message.
	m, err = it.Next()
	if m != nil {
		t.Errorf("message got: %v ; want: nil", m)
	}
	if err != Done {
		t.Errorf("err got: %v ; want: %v", err, Done)
	}
}
