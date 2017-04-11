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

	"golang.org/x/net/context"
)

const MaxPublishBatchSize = 1000

// Topic is a reference to a PubSub topic.
type Topic struct {
	s service

	// The fully qualified identifier for the topic, in the format "projects/<projid>/topics/<name>"
	name string
}

// CreateTopic creates a new topic.
// The specified topic name must start with a letter, and contain only letters
// ([A-Za-z]), numbers ([0-9]), dashes (-), underscores (_), periods (.),
// tildes (~), plus (+) or percent signs (%). It must be between 3 and 255
// characters in length, and must not start with "goog".
// If the topic already exists an error will be returned.
func (c *Client) CreateTopic(ctx context.Context, name string) (*Topic, error) {
	t := c.Topic(name)
	err := c.s.createTopic(ctx, t.Name())
	return t, err
}

// Topic creates a reference to a topic.
func (c *Client) Topic(name string) *Topic {
	return &Topic{s: c.s, name: fmt.Sprintf("projects/%s/topics/%s", c.projectID, name)}
}

// Topics returns an iterator which returns all of the topics for the client's project.
func (c *Client) Topics(ctx context.Context) *TopicIterator {
	return &TopicIterator{
		s: c.s,
		stringsIterator: stringsIterator{
			ctx: ctx,
			fetch: func(ctx context.Context, tok string) (*stringsPage, error) {
				return c.s.listProjectTopics(ctx, c.fullyQualifiedProjectName(), tok)
			},
		},
	}
}

// TopicIterator is an iterator that returns a series of topics.
type TopicIterator struct {
	s service
	stringsIterator
}

// Next returns the next topic. If there are no more topics, Done will be returned.
func (tps *TopicIterator) Next() (*Topic, error) {
	topicName, err := tps.stringsIterator.Next()
	if err != nil {
		return nil, err
	}
	return &Topic{s: tps.s, name: topicName}, nil
}

// Name returns the globally unique name for the topic.
func (t *Topic) Name() string {
	return t.name
}

// Delete deletes the topic.
func (t *Topic) Delete(ctx context.Context) error {
	return t.s.deleteTopic(ctx, t.name)
}

// Exists reports whether the topic exists on the server.
func (t *Topic) Exists(ctx context.Context) (bool, error) {
	if t.name == "_deleted-topic_" {
		return false, nil
	}

	return t.s.topicExists(ctx, t.name)
}

// Subscriptions returns an iterator which returns the subscriptions for this topic.
func (t *Topic) Subscriptions(ctx context.Context) *SubscriptionIterator {
	// NOTE: zero or more Subscriptions that are ultimately returned by this
	// Subscriptions iterator may belong to a different project to t.
	return &SubscriptionIterator{
		s: t.s,
		stringsIterator: stringsIterator{
			ctx: ctx,
			fetch: func(ctx context.Context, tok string) (*stringsPage, error) {

				return t.s.listTopicSubscriptions(ctx, t.name, tok)
			},
		},
	}
}

// Publish publishes the supplied Messages to the topic.
// If successful, the server-assigned message IDs are returned in the same order as the supplied Messages.
// At most MaxPublishBatchSize messages may be supplied.
func (t *Topic) Publish(ctx context.Context, msgs ...*Message) ([]string, error) {
	if len(msgs) == 0 {
		return nil, nil
	}
	if len(msgs) > MaxPublishBatchSize {
		return nil, fmt.Errorf("pubsub: got %d messages, but maximum batch size is %d", len(msgs), MaxPublishBatchSize)
	}
	return t.s.publishMessages(ctx, t.name, msgs)
}
