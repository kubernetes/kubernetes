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

type topicListCall struct {
	inTok, outTok string
	topics        []string
	err           error
}

type topicListService struct {
	service
	calls []topicListCall

	t *testing.T // for error logging.
}

func (s *topicListService) listProjectTopics(ctx context.Context, projName, pageTok string) (*stringsPage, error) {
	if len(s.calls) == 0 || projName != "projects/projid" {
		s.t.Errorf("unexpected call: projName: %q, pageTok: %q", projName, pageTok)
		return nil, errors.New("bang")
	}

	call := s.calls[0]
	s.calls = s.calls[1:]
	if call.inTok != pageTok {
		s.t.Errorf("page token: got: %v, want: %v", pageTok, call.inTok)
	}
	return &stringsPage{call.topics, call.outTok}, call.err
}

func checkTopicListing(t *testing.T, calls []topicListCall, want []string) {
	s := &topicListService{calls: calls, t: t}
	c := &Client{projectID: "projid", s: s}
	topics, err := slurpTopics(c.Topics(context.Background()))
	if err != nil {
		t.Errorf("error listing topics: %v", err)
	}
	got := topicNames(topics)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("topic list: got: %v, want: %v", got, want)
	}
	if len(s.calls) != 0 {
		t.Errorf("outstanding calls: %v", s.calls)
	}
}

// All returns the remaining topics from this iterator.
func slurpTopics(it *TopicIterator) ([]*Topic, error) {
	var topics []*Topic
	for {
		switch topic, err := it.Next(); err {
		case nil:
			topics = append(topics, topic)
		case Done:
			return topics, nil
		default:
			return nil, err
		}
	}
}

func TestListTopics(t *testing.T) {
	calls := []topicListCall{
		{
			topics: []string{"t1", "t2"},
			outTok: "a",
		},
		{
			inTok:  "a",
			topics: []string{"t3"},
			outTok: "b",
		},
		{
			inTok:  "b",
			topics: []string{},
			outTok: "c",
		},
		{
			inTok:  "c",
			topics: []string{"t4"},
			outTok: "",
		},
	}
	checkTopicListing(t, calls, []string{"t1", "t2", "t3", "t4"})
}

func TestListCompletelyEmptyTopics(t *testing.T) {
	calls := []topicListCall{
		{
			outTok: "",
		},
	}
	var want []string
	checkTopicListing(t, calls, want)
}

func TestListFinalEmptyPage(t *testing.T) {
	calls := []topicListCall{
		{
			topics: []string{"t1", "t2"},
			outTok: "a",
		},
		{
			inTok:  "a",
			topics: []string{},
			outTok: "",
		},
	}
	checkTopicListing(t, calls, []string{"t1", "t2"})
}

func topicNames(topics []*Topic) []string {
	var names []string

	for _, topic := range topics {
		names = append(names, topic.name)

	}
	return names
}
