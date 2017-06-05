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

type subListCall struct {
	inTok, outTok string
	subs          []string
	err           error
}

type subListService struct {
	service
	calls []subListCall

	t *testing.T // for error logging.
}

func (s *subListService) listSubs(pageTok string) (*stringsPage, error) {
	if len(s.calls) == 0 {
		s.t.Errorf("unexpected call: pageTok: %q", pageTok)
		return nil, errors.New("bang")
	}

	call := s.calls[0]
	s.calls = s.calls[1:]
	if call.inTok != pageTok {
		s.t.Errorf("page token: got: %v, want: %v", pageTok, call.inTok)
	}
	return &stringsPage{call.subs, call.outTok}, call.err
}

func (s *subListService) listProjectSubscriptions(ctx context.Context, projName, pageTok string) (*stringsPage, error) {
	if projName != "projects/projid" {
		s.t.Errorf("unexpected call: projName: %q, pageTok: %q", projName, pageTok)
		return nil, errors.New("bang")
	}
	return s.listSubs(pageTok)
}

func (s *subListService) listTopicSubscriptions(ctx context.Context, topicName, pageTok string) (*stringsPage, error) {
	if topicName != "projects/projid/topics/topic" {
		s.t.Errorf("unexpected call: topicName: %q, pageTok: %q", topicName, pageTok)
		return nil, errors.New("bang")
	}
	return s.listSubs(pageTok)
}

// All returns the remaining subscriptions from this iterator.
func slurpSubs(it *SubscriptionIterator) ([]*Subscription, error) {
	var subs []*Subscription
	for {
		switch sub, err := it.Next(); err {
		case nil:
			subs = append(subs, sub)
		case Done:
			return subs, nil
		default:
			return nil, err
		}
	}
}

func TestListProjectSubscriptions(t *testing.T) {
	calls := []subListCall{
		{
			subs:   []string{"s1", "s2"},
			outTok: "a",
		},
		{
			inTok:  "a",
			subs:   []string{"s3"},
			outTok: "",
		},
	}
	s := &subListService{calls: calls, t: t}
	c := &Client{projectID: "projid", s: s}
	subs, err := slurpSubs(c.Subscriptions(context.Background()))
	if err != nil {
		t.Errorf("error listing subscriptions: %v", err)
	}
	got := subNames(subs)
	want := []string{"s1", "s2", "s3"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("sub list: got: %v, want: %v", got, want)
	}
	if len(s.calls) != 0 {
		t.Errorf("outstanding calls: %v", s.calls)
	}
}

func TestListTopicSubscriptions(t *testing.T) {
	calls := []subListCall{
		{
			subs:   []string{"s1", "s2"},
			outTok: "a",
		},
		{
			inTok:  "a",
			subs:   []string{"s3"},
			outTok: "",
		},
	}
	s := &subListService{calls: calls, t: t}
	c := &Client{projectID: "projid", s: s}
	subs, err := slurpSubs(c.Topic("topic").Subscriptions(context.Background()))
	if err != nil {
		t.Errorf("error listing subscriptions: %v", err)
	}
	got := subNames(subs)
	want := []string{"s1", "s2", "s3"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("sub list: got: %v, want: %v", got, want)
	}
	if len(s.calls) != 0 {
		t.Errorf("outstanding calls: %v", s.calls)
	}
}

func subNames(subs []*Subscription) []string {
	var names []string

	for _, sub := range subs {
		names = append(names, sub.name)

	}
	return names
}
