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

type fetcherService struct {
	service
	msgs [][]*Message
}

func (s *fetcherService) fetchMessages(ctx context.Context, subName string, maxMessages int64) ([]*Message, error) {
	if len(s.msgs) == 0 {
		return nil, errors.New("bang")
	}
	ret := s.msgs[0]
	s.msgs = s.msgs[1:]
	return ret, nil
}

func TestPuller(t *testing.T) {
	s := &fetcherService{
		msgs: [][]*Message{
			{{AckID: "a"}, {AckID: "b"}},
			{},
			{{AckID: "c"}, {AckID: "d"}},
			{{AckID: "e"}},
		},
	}
	c := &Client{projectID: "projid", s: s}

	pulled := make(chan string, 10)
	pull := &puller{
		Client:    c,
		Sub:       "subname",
		BatchSize: 2,
		Notify:    func(ackID string) { pulled <- ackID },
	}

	got := []string{}
	for i := 0; i < 5; i++ {
		m, err := pull.Next(context.Background())
		got = append(got, m.AckID)
		if err != nil {
			t.Errorf("unexpected err from pull.Next: %v", err)
		}
	}
	_, err := pull.Next(context.Background())
	if err == nil {
		t.Errorf("unexpected err from pull.Next: %v", err)
	}

	want := []string{"a", "b", "c", "d", "e"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("pulled ack ids: got: %v ; want: %v", got, want)
	}
}

func TestPullerNotification(t *testing.T) {
	s := &fetcherService{
		msgs: [][]*Message{
			{{AckID: "a"}, {AckID: "b"}},
			{{AckID: "c"}, {AckID: "d"}},
		},
	}
	c := &Client{projectID: "projid", s: s}

	pulled := make(chan string, 10)
	pull := &puller{
		Client:    c,
		Sub:       "subname",
		BatchSize: 2,
		Notify:    func(ackID string) { pulled <- ackID },
	}

	got := []string{}
	for i := 0; i < 3; i++ {
		m, err := pull.Next(context.Background())
		got = append(got, m.AckID)
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
