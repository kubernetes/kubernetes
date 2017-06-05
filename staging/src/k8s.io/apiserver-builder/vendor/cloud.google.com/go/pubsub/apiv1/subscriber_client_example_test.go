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

// AUTO-GENERATED CODE. DO NOT EDIT.

package pubsub_test

import (
	"cloud.google.com/go/pubsub/apiv1"
	"golang.org/x/net/context"
	pubsubpb "google.golang.org/genproto/googleapis/pubsub/v1"
)

func ExampleNewSubscriberClient() {
	ctx := context.Background()
	c, err := pubsub.NewSubscriberClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use client.
	_ = c
}

func ExampleSubscriberClient_CreateSubscription() {
	ctx := context.Background()
	c, err := pubsub.NewSubscriberClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &pubsubpb.Subscription{
	// TODO: Fill request struct fields.
	}
	resp, err := c.CreateSubscription(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleSubscriberClient_GetSubscription() {
	ctx := context.Background()
	c, err := pubsub.NewSubscriberClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &pubsubpb.GetSubscriptionRequest{
	// TODO: Fill request struct fields.
	}
	resp, err := c.GetSubscription(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleSubscriberClient_ListSubscriptions() {
	ctx := context.Background()
	c, err := pubsub.NewSubscriberClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &pubsubpb.ListSubscriptionsRequest{
	// TODO: Fill request struct fields.
	}
	it := c.ListSubscriptions(ctx, req)
	for {
		resp, err := it.Next()
		if err != nil {
			// TODO: Handle error.
			break
		}
		// TODO: Use resp.
		_ = resp
	}
}

func ExampleSubscriberClient_DeleteSubscription() {
	ctx := context.Background()
	c, err := pubsub.NewSubscriberClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &pubsubpb.DeleteSubscriptionRequest{
	// TODO: Fill request struct fields.
	}
	err = c.DeleteSubscription(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
}

func ExampleSubscriberClient_ModifyAckDeadline() {
	ctx := context.Background()
	c, err := pubsub.NewSubscriberClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &pubsubpb.ModifyAckDeadlineRequest{
	// TODO: Fill request struct fields.
	}
	err = c.ModifyAckDeadline(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
}

func ExampleSubscriberClient_Acknowledge() {
	ctx := context.Background()
	c, err := pubsub.NewSubscriberClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &pubsubpb.AcknowledgeRequest{
	// TODO: Fill request struct fields.
	}
	err = c.Acknowledge(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
}

func ExampleSubscriberClient_Pull() {
	ctx := context.Background()
	c, err := pubsub.NewSubscriberClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &pubsubpb.PullRequest{
	// TODO: Fill request struct fields.
	}
	resp, err := c.Pull(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleSubscriberClient_ModifyPushConfig() {
	ctx := context.Background()
	c, err := pubsub.NewSubscriberClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &pubsubpb.ModifyPushConfigRequest{
	// TODO: Fill request struct fields.
	}
	err = c.ModifyPushConfig(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
}
