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
	"testing"

	"golang.org/x/net/context"
)

type fetchResult struct {
	msgs []*Message
	err  error
}

type fetcherService struct {
	service
	results        []fetchResult
	unexpectedCall bool
}

func (s *fetcherService) fetchMessages(ctx context.Context, subName string, maxMessages int64) ([]*Message, error) {
	if len(s.results) == 0 {
		s.unexpectedCall = true
		return nil, errors.New("bang")
	}
	ret := s.results[0]
	s.results = s.results[1:]
	return ret.msgs, ret.err
}

func TestPuller(t *testing.T) {
	s := &fetcherService{
		results: []fetchResult{
			{
				msgs: []*Message{{ackID: "a"}, {ackID: "b"}},
			},
			{},
			{
				msgs: []*Message{{ackID: "c"}, {ackID: "d"}},
			},
			{
				msgs: []*Message{{ackID: "e"}},
			},
		},
	}

	pulled := make(chan string, 10)

	pull := newPuller(s, "subname", context.Background(), 2, func(ackID string) { pulled <- ackID }, func(string) {})

	got := []string{}
	for i := 0; i < 5; i++ {
		m, err := pull.Next()
		got = append(got, m.ackID)
		if err != nil {
			t.Errorf("unexpected err from pull.Next: %v", err)
		}
	}
	_, err := pull.Next()
	if err == nil {
		t.Errorf("unexpected err from pull.Next: %v", err)
	}

	want := []string{"a", "b", "c", "d", "e"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("pulled ack ids: got: %v ; want: %v", got, want)
	}
}

func TestPullerAddsToKeepAlive(t *testing.T) {
	s := &fetcherService{
		results: []fetchResult{
			{
				msgs: []*Message{{ackID: "a"}, {ackID: "b"}},
			},
			{
				msgs: []*Message{{ackID: "c"}, {ackID: "d"}},
			},
		},
	}

	pulled := make(chan string, 10)

	pull := newPuller(s, "subname", context.Background(), 2, func(ackID string) { pulled <- ackID }, func(string) {})

	got := []string{}
	for i := 0; i < 3; i++ {
		m, err := pull.Next()
		got = append(got, m.ackID)
		if err != nil {
			t.Errorf("unexpected err from pull.Next: %v", err)
		}
	}

	want := []string{"a", "b", "c"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("pulled ack ids: got: %v ; want: %v", got, want)
	}

	close(pulled)
	// We should have seen "d" written to the channel too, even though it hasn't been returned yet.
	pulledIDs := []string{}
	for id := range pulled {
		pulledIDs = append(pulledIDs, id)
	}

	want = append(want, "d")
	if !reflect.DeepEqual(pulledIDs, want) {
		t.Errorf("pulled ack ids: got: %v ; want: %v", pulledIDs, want)
	}
}

func TestPullerRetriesOnce(t *testing.T) {
	bang := errors.New("bang")
	s := &fetcherService{
		results: []fetchResult{
			{
				err: bang,
			},
			{
				err: bang,
			},
		},
	}

	pull := newPuller(s, "subname", context.Background(), 2, func(string) {}, func(string) {})

	_, err := pull.Next()
	if err != bang {
		t.Errorf("pull.Next err got: %v, want: %v", err, bang)
	}

	if s.unexpectedCall {
		t.Errorf("unexpected retry")
	}
	if len(s.results) != 0 {
		t.Errorf("outstanding calls: got: %v, want: 0", len(s.results))
	}
}
