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
	"encoding/base64"
	"fmt"
	"net/http"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/api/googleapi"
	raw "google.golang.org/api/pubsub/v1"
)

// service provides an internal abstraction to isolate the generated
// PubSub API; most of this package uses this interface instead.
// The single implementation, *apiService, contains all the knowledge
// of the generated PubSub API (except for that present in legacy code).
type service interface {
	createSubscription(ctx context.Context, topicName, subName string, ackDeadline time.Duration, pushConfig *PushConfig) error
	getSubscriptionConfig(ctx context.Context, subName string) (*SubscriptionConfig, string, error)
	listProjectSubscriptions(ctx context.Context, projName, pageTok string) (*stringsPage, error)
	deleteSubscription(ctx context.Context, name string) error
	subscriptionExists(ctx context.Context, name string) (bool, error)
	modifyPushConfig(ctx context.Context, subName string, conf *PushConfig) error

	createTopic(ctx context.Context, name string) error
	deleteTopic(ctx context.Context, name string) error
	topicExists(ctx context.Context, name string) (bool, error)
	listProjectTopics(ctx context.Context, projName, pageTok string) (*stringsPage, error)
	listTopicSubscriptions(ctx context.Context, topicName, pageTok string) (*stringsPage, error)

	modifyAckDeadline(ctx context.Context, subName string, deadline time.Duration, ackIDs []string) error
	fetchMessages(ctx context.Context, subName string, maxMessages int64) ([]*Message, error)
	publishMessages(ctx context.Context, topicName string, msgs []*Message) ([]string, error)

	// splitAckIDs divides ackIDs into
	//  * a batch of a size which is suitable for passing to acknowledge or
	//    modifyAckDeadline, and
	//  * the rest.
	splitAckIDs(ackIDs []string) ([]string, []string)

	// acknowledge ACKs the IDs in ackIDs.
	acknowledge(ctx context.Context, subName string, ackIDs []string) error
}

type apiService struct {
	s *raw.Service
}

func newPubSubService(client *http.Client, endpoint string) (*apiService, error) {
	s, err := raw.New(client)
	if err != nil {
		return nil, err
	}
	s.BasePath = endpoint

	return &apiService{s: s}, nil
}

func (s *apiService) createSubscription(ctx context.Context, topicName, subName string, ackDeadline time.Duration, pushConfig *PushConfig) error {
	var rawPushConfig *raw.PushConfig
	if pushConfig != nil {
		rawPushConfig = &raw.PushConfig{
			Attributes:   pushConfig.Attributes,
			PushEndpoint: pushConfig.Endpoint,
		}
	}
	rawSub := &raw.Subscription{
		AckDeadlineSeconds: int64(ackDeadline.Seconds()),
		PushConfig:         rawPushConfig,
		Topic:              topicName,
	}
	_, err := s.s.Projects.Subscriptions.Create(subName, rawSub).Context(ctx).Do()
	return err
}

func (s *apiService) getSubscriptionConfig(ctx context.Context, subName string) (*SubscriptionConfig, string, error) {
	rawSub, err := s.s.Projects.Subscriptions.Get(subName).Context(ctx).Do()
	if err != nil {
		return nil, "", err
	}
	sub := &SubscriptionConfig{
		AckDeadline: time.Second * time.Duration(rawSub.AckDeadlineSeconds),
		PushConfig: PushConfig{
			Endpoint:   rawSub.PushConfig.PushEndpoint,
			Attributes: rawSub.PushConfig.Attributes,
		},
	}
	return sub, rawSub.Topic, err
}

// stringsPage contains a list of strings and a token for fetching the next page.
type stringsPage struct {
	strings []string
	tok     string
}

func (s *apiService) listProjectSubscriptions(ctx context.Context, projName, pageTok string) (*stringsPage, error) {
	resp, err := s.s.Projects.Subscriptions.List(projName).PageToken(pageTok).Context(ctx).Do()
	if err != nil {
		return nil, err
	}
	subs := []string{}
	for _, sub := range resp.Subscriptions {
		subs = append(subs, sub.Name)
	}
	return &stringsPage{subs, resp.NextPageToken}, nil
}

func (s *apiService) deleteSubscription(ctx context.Context, name string) error {
	_, err := s.s.Projects.Subscriptions.Delete(name).Context(ctx).Do()
	return err
}

func (s *apiService) subscriptionExists(ctx context.Context, name string) (bool, error) {
	_, err := s.s.Projects.Subscriptions.Get(name).Context(ctx).Do()
	if err == nil {
		return true, nil
	}
	if e, ok := err.(*googleapi.Error); ok && e.Code == http.StatusNotFound {
		return false, nil
	}
	return false, err
}

func (s *apiService) createTopic(ctx context.Context, name string) error {
	// Note: The raw API expects a Topic body, but ignores it.
	_, err := s.s.Projects.Topics.Create(name, &raw.Topic{}).
		Context(ctx).
		Do()
	return err
}

func (s *apiService) listProjectTopics(ctx context.Context, projName, pageTok string) (*stringsPage, error) {
	resp, err := s.s.Projects.Topics.List(projName).PageToken(pageTok).Context(ctx).Do()
	if err != nil {
		return nil, err
	}
	topics := []string{}
	for _, topic := range resp.Topics {
		topics = append(topics, topic.Name)
	}
	return &stringsPage{topics, resp.NextPageToken}, nil
}

func (s *apiService) deleteTopic(ctx context.Context, name string) error {
	_, err := s.s.Projects.Topics.Delete(name).Context(ctx).Do()
	return err
}

func (s *apiService) topicExists(ctx context.Context, name string) (bool, error) {
	_, err := s.s.Projects.Topics.Get(name).Context(ctx).Do()
	if err == nil {
		return true, nil
	}
	if e, ok := err.(*googleapi.Error); ok && e.Code == http.StatusNotFound {
		return false, nil
	}
	return false, err
}

func (s *apiService) listTopicSubscriptions(ctx context.Context, topicName, pageTok string) (*stringsPage, error) {
	resp, err := s.s.Projects.Topics.Subscriptions.List(topicName).PageToken(pageTok).Context(ctx).Do()
	if err != nil {
		return nil, err
	}
	subs := []string{}
	for _, sub := range resp.Subscriptions {
		subs = append(subs, sub)
	}
	return &stringsPage{subs, resp.NextPageToken}, nil
}

func (s *apiService) modifyAckDeadline(ctx context.Context, subName string, deadline time.Duration, ackIDs []string) error {
	req := &raw.ModifyAckDeadlineRequest{
		AckDeadlineSeconds: int64(deadline.Seconds()),
		AckIds:             ackIDs,
	}
	_, err := s.s.Projects.Subscriptions.ModifyAckDeadline(subName, req).
		Context(ctx).
		Do()
	return err
}

// maxPayload is the maximum number of bytes to devote to actual ids in
// acknowledgement or modifyAckDeadline requests.  Note that there is ~1K of
// constant overhead, plus 3 bytes per ID (two quotes and a comma).  The total
// payload size may not exceed 512K.
const maxPayload = 500 * 1024
const overheadPerID = 3 // 3 bytes of JSON

// splitAckIDs splits ids into two slices, the first of which contains at most maxPayload bytes of ackID data.
func (s *apiService) splitAckIDs(ids []string) ([]string, []string) {
	total := 0
	for i, id := range ids {
		total += len(id) + overheadPerID
		if total > maxPayload {
			return ids[:i], ids[i:]
		}
	}
	return ids, nil
}

func (s *apiService) acknowledge(ctx context.Context, subName string, ackIDs []string) error {
	req := &raw.AcknowledgeRequest{
		AckIds: ackIDs,
	}
	_, err := s.s.Projects.Subscriptions.Acknowledge(subName, req).
		Context(ctx).
		Do()
	return err
}

func (s *apiService) fetchMessages(ctx context.Context, subName string, maxMessages int64) ([]*Message, error) {
	req := &raw.PullRequest{
		MaxMessages: maxMessages,
	}
	resp, err := s.s.Projects.Subscriptions.Pull(subName, req).
		Context(ctx).
		Do()

	if err != nil {
		return nil, err
	}

	msgs := make([]*Message, 0, len(resp.ReceivedMessages))
	for i, m := range resp.ReceivedMessages {
		msg, err := toMessage(m)
		if err != nil {
			return nil, fmt.Errorf("pubsub: cannot decode the retrieved message at index: %d, message: %+v", i, m)
		}
		msgs = append(msgs, msg)
	}

	return msgs, nil
}

func (s *apiService) publishMessages(ctx context.Context, topicName string, msgs []*Message) ([]string, error) {
	rawMsgs := make([]*raw.PubsubMessage, len(msgs))
	for i, msg := range msgs {
		rawMsgs[i] = &raw.PubsubMessage{
			Data:       base64.StdEncoding.EncodeToString(msg.Data),
			Attributes: msg.Attributes,
		}
	}

	req := &raw.PublishRequest{Messages: rawMsgs}
	resp, err := s.s.Projects.Topics.Publish(topicName, req).
		Context(ctx).
		Do()

	if err != nil {
		return nil, err
	}
	return resp.MessageIds, nil
}

func (s *apiService) modifyPushConfig(ctx context.Context, subName string, conf *PushConfig) error {
	req := &raw.ModifyPushConfigRequest{
		PushConfig: &raw.PushConfig{
			Attributes:   conf.Attributes,
			PushEndpoint: conf.Endpoint,
		},
	}
	_, err := s.s.Projects.Subscriptions.ModifyPushConfig(subName, req).
		Context(ctx).
		Do()
	return err
}
