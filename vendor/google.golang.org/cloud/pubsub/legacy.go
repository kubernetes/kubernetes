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
	"errors"
	"fmt"
	"net/http"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/api/googleapi"
	raw "google.golang.org/api/pubsub/v1"
	"google.golang.org/cloud/internal"
)

// batchLimit is maximun size of a single batch.
const batchLimit = 1000

// CreateTopic creates a new topic with the specified name on the backend.
//
// Deprecated: Use Client.NewTopic instead.
//
// It will return an error if topic already exists.
func CreateTopic(ctx context.Context, name string) error {
	_, err := rawService(ctx).Projects.Topics.Create(fullTopicName(internal.ProjID(ctx), name), &raw.Topic{}).Do()
	return err
}

// DeleteTopic deletes the specified topic.
//
// Deprecated: Use TopicHandle.Delete instead.
func DeleteTopic(ctx context.Context, name string) error {
	_, err := rawService(ctx).Projects.Topics.Delete(fullTopicName(internal.ProjID(ctx), name)).Do()
	return err
}

// TopicExists returns true if a topic exists with the specified name.
//
// Deprecated: Use TopicHandle.Exists instead.
func TopicExists(ctx context.Context, name string) (bool, error) {
	_, err := rawService(ctx).Projects.Topics.Get(fullTopicName(internal.ProjID(ctx), name)).Do()
	if e, ok := err.(*googleapi.Error); ok && e.Code == http.StatusNotFound {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// DeleteSub deletes the subscription.
//
// Deprecated: Use SubscriptionHandle.Delete instead.
func DeleteSub(ctx context.Context, name string) error {
	_, err := rawService(ctx).Projects.Subscriptions.Delete(fullSubName(internal.ProjID(ctx), name)).Do()
	return err
}

// SubExists returns true if subscription exists.
//
// Deprecated: Use SubscriptionHandle.Exists instead.
func SubExists(ctx context.Context, name string) (bool, error) {
	_, err := rawService(ctx).Projects.Subscriptions.Get(fullSubName(internal.ProjID(ctx), name)).Do()
	if e, ok := err.(*googleapi.Error); ok && e.Code == http.StatusNotFound {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// CreateSub creates a Pub/Sub subscription on the backend.
//
// Deprecated: Use TopicHandle.Subscribe instead.
//
// A subscription should subscribe to an existing topic.
//
// The messages that haven't acknowledged will be pushed back to the
// subscription again when the default acknowledgement deadline is
// reached. You can override the default deadline by providing a
// non-zero deadline. Deadline must not be specified to
// precision greater than one second.
//
// As new messages are being queued on the subscription, you
// may recieve push notifications regarding to the new arrivals.
// To receive notifications of new messages in the queue,
// specify an endpoint callback URL.
// If endpoint is an empty string the backend will not notify the
// client of new messages.
//
// If the subscription already exists an error will be returned.
func CreateSub(ctx context.Context, name string, topic string, deadline time.Duration, endpoint string) error {
	sub := &raw.Subscription{
		Topic: fullTopicName(internal.ProjID(ctx), topic),
	}
	if int64(deadline) > 0 {
		if !isSec(deadline) {
			return errors.New("pubsub: deadline must not be specified to precision greater than one second")
		}
		sub.AckDeadlineSeconds = int64(deadline / time.Second)
	}
	if endpoint != "" {
		sub.PushConfig = &raw.PushConfig{PushEndpoint: endpoint}
	}
	_, err := rawService(ctx).Projects.Subscriptions.Create(fullSubName(internal.ProjID(ctx), name), sub).Do()
	return err
}

// Pull pulls up to n messages from the subscription. n must not be larger than 100.
//
// Deprecated: Use Subscription.Pull instead
func Pull(ctx context.Context, sub string, n int) ([]*Message, error) {
	return pull(ctx, sub, n, true)
}

// PullWait pulls up to n messages from the subscription. If there are no
// messages in the queue, it will wait until at least one message is
// available or a timeout occurs. n must not be larger than 100.
//
// Deprecated: Use Subscription.Pull instead
func PullWait(ctx context.Context, sub string, n int) ([]*Message, error) {
	return pull(ctx, sub, n, false)
}

func pull(ctx context.Context, sub string, n int, retImmediately bool) ([]*Message, error) {
	if n < 1 || n > batchLimit {
		return nil, fmt.Errorf("pubsub: cannot pull less than one, more than %d messages, but %d was given", batchLimit, n)
	}
	resp, err := rawService(ctx).Projects.Subscriptions.Pull(fullSubName(internal.ProjID(ctx), sub), &raw.PullRequest{
		ReturnImmediately: retImmediately,
		MaxMessages:       int64(n),
	}).Do()
	if err != nil {
		return nil, err
	}
	msgs := make([]*Message, len(resp.ReceivedMessages))
	for i := 0; i < len(resp.ReceivedMessages); i++ {
		msg, err := toMessage(resp.ReceivedMessages[i])
		if err != nil {
			return nil, fmt.Errorf("pubsub: cannot decode the retrieved message at index: %d, PullResponse: %+v", i, resp.ReceivedMessages[i])
		}
		msgs[i] = msg
	}
	return msgs, nil
}

// ModifyAckDeadline modifies the acknowledgement deadline
// for the messages retrieved from the specified subscription.
// Deadline must not be specified to precision greater than one second.
//
// Deprecated: Use Subscription.Pull instead, which automatically extends ack deadlines.
func ModifyAckDeadline(ctx context.Context, sub string, id string, deadline time.Duration) error {
	if !isSec(deadline) {
		return errors.New("pubsub: deadline must not be specified to precision greater than one second")
	}
	_, err := rawService(ctx).Projects.Subscriptions.ModifyAckDeadline(fullSubName(internal.ProjID(ctx), sub), &raw.ModifyAckDeadlineRequest{
		AckDeadlineSeconds: int64(deadline / time.Second),
		AckIds:             []string{id},
	}).Do()
	return err
}

// Ack acknowledges one or more Pub/Sub messages on the
// specified subscription.
//
// Deprecated: Call Message.Done on a Message returned by Iterator.Next instead.
func Ack(ctx context.Context, sub string, id ...string) error {
	for idx, ackID := range id {
		if ackID == "" {
			return fmt.Errorf("pubsub: empty ackID detected at index %d", idx)
		}
	}
	_, err := rawService(ctx).Projects.Subscriptions.Acknowledge(fullSubName(internal.ProjID(ctx), sub), &raw.AcknowledgeRequest{
		AckIds: id,
	}).Do()
	return err
}

func isSec(dur time.Duration) bool {
	return dur%time.Second == 0
}

// Publish publishes messages to the topic's subscribers. It returns
// message IDs upon success.
//
// Deprecated: Use TopicHandle.Publish instead.
func Publish(ctx context.Context, topic string, msgs ...*Message) ([]string, error) {
	var rawMsgs []*raw.PubsubMessage
	if len(msgs) == 0 {
		return nil, errors.New("pubsub: no messages to publish")
	}
	if len(msgs) > batchLimit {
		return nil, fmt.Errorf("pubsub: %d messages given, but maximum batch size is %d", len(msgs), batchLimit)
	}
	rawMsgs = make([]*raw.PubsubMessage, len(msgs))
	for i, msg := range msgs {
		rawMsgs[i] = &raw.PubsubMessage{
			Data:       base64.StdEncoding.EncodeToString(msg.Data),
			Attributes: msg.Attributes,
		}
	}
	resp, err := rawService(ctx).Projects.Topics.Publish(fullTopicName(internal.ProjID(ctx), topic), &raw.PublishRequest{
		Messages: rawMsgs,
	}).Do()
	if err != nil {
		return nil, err
	}
	return resp.MessageIds, nil
}

// ModifyPushEndpoint modifies the URL endpoint to modify the resource
// to handle push notifications coming from the Pub/Sub backend
// for the specified subscription.
//
// Deprecated: Use SubscriptionHandle.ModifyPushConfig instead.
func ModifyPushEndpoint(ctx context.Context, sub, endpoint string) error {
	_, err := rawService(ctx).Projects.Subscriptions.ModifyPushConfig(fullSubName(internal.ProjID(ctx), sub), &raw.ModifyPushConfigRequest{
		PushConfig: &raw.PushConfig{
			PushEndpoint: endpoint,
		},
	}).Do()
	return err
}

// fullSubName returns the fully qualified name for a subscription.
// E.g. /subscriptions/project-id/subscription-name.
func fullSubName(proj, name string) string {
	return fmt.Sprintf("projects/%s/subscriptions/%s", proj, name)
}

func rawService(ctx context.Context) *raw.Service {
	return internal.Service(ctx, "pubsub", func(hc *http.Client) interface{} {
		svc, _ := raw.New(hc)
		svc.BasePath = baseAddr()
		return svc
	}).(*raw.Service)
}

// fullTopicName returns the fully qualified name for a topic.
// E.g. /topics/project-id/topic-name.
func fullTopicName(proj, name string) string {
	return fmt.Sprintf("projects/%s/topics/%s", proj, name)
}
