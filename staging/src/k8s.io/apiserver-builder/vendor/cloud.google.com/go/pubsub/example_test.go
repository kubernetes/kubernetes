// Copyright 2014 Google Inc. All Rights Reserved.
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

package pubsub_test

import (
	"fmt"
	"time"

	"cloud.google.com/go/pubsub"
	"golang.org/x/net/context"
)

func ExampleNewClient() {
	ctx := context.Background()
	_, err := pubsub.NewClient(ctx, "project-id")
	if err != nil {
		// TODO: Handle error.
	}

	// See the other examples to learn how to use the Client.
}

func ExampleClient_CreateTopic() {
	ctx := context.Background()
	client, err := pubsub.NewClient(ctx, "project-id")
	if err != nil {
		// TODO: Handle error.
	}

	// Create a new topic with the given name.
	topic, err := client.CreateTopic(ctx, "topicName")
	if err != nil {
		// TODO: Handle error.
	}

	_ = topic // TODO: use the topic.
}

func ExampleClient_Topics() {
	ctx := context.Background()
	client, err := pubsub.NewClient(ctx, "project-id")
	if err != nil {
		// TODO: Handle error.
	}
	// List all topics.
	it := client.Topics(ctx)
	_ = it // See the TopicIterator example for its usage.
}

func ExampleClient_CreateSubscription() {
	ctx := context.Background()
	client, err := pubsub.NewClient(ctx, "project-id")
	if err != nil {
		// TODO: Handle error.
	}

	// Create a new topic with the given name.
	topic, err := client.CreateTopic(ctx, "topicName")
	if err != nil {
		// TODO: Handle error.
	}

	// Create a new subscription to the previously created topic
	// with the given name.
	sub, err := client.CreateSubscription(ctx, "subName", topic, 10*time.Second, nil)
	if err != nil {
		// TODO: Handle error.
	}

	_ = sub // TODO: use the subscription.
}

func ExampleClient_Subscriptions() {
	ctx := context.Background()
	client, err := pubsub.NewClient(ctx, "project-id")
	if err != nil {
		// TODO: Handle error.
	}
	// List all subscriptions of the project.
	it := client.Subscriptions(ctx)
	_ = it // See the SubscriptionIterator example for its usage.
}

func ExampleTopic_Delete() {
	ctx := context.Background()
	client, err := pubsub.NewClient(ctx, "project-id")
	if err != nil {
		// TODO: Handle error.
	}

	topic := client.Topic("topicName")
	if err := topic.Delete(ctx); err != nil {
		// TODO: Handle error.
	}
}

func ExampleTopic_Exists() {
	ctx := context.Background()
	client, err := pubsub.NewClient(ctx, "project-id")
	if err != nil {
		// TODO: Handle error.
	}

	topic := client.Topic("topicName")
	ok, err := topic.Exists(ctx)
	if err != nil {
		// TODO: Handle error.
	}
	if !ok {
		// Topic doesn't exist.
	}
}

func ExampleTopic_Publish() {
	ctx := context.Background()
	client, err := pubsub.NewClient(ctx, "project-id")
	if err != nil {
		// TODO: Handle error.
	}

	topic := client.Topic("topicName")
	msgIDs, err := topic.Publish(ctx, &pubsub.Message{
		Data: []byte("hello world"),
	})
	if err != nil {
		// TODO: Handle error.
	}
	fmt.Printf("Published a message with a message ID: %s\n", msgIDs[0])
}

func ExampleTopic_Subscriptions() {
	ctx := context.Background()
	client, err := pubsub.NewClient(ctx, "project-id")
	if err != nil {
		// TODO: Handle error.
	}
	topic := client.Topic("topic-name")
	// List all subscriptions of the topic (maybe of multiple projects).
	for subs := topic.Subscriptions(ctx); ; {
		sub, err := subs.Next()
		if err == pubsub.Done {
			break
		}
		if err != nil {
			// TODO: Handle error.
		}
		_ = sub // TODO: use the subscription.
	}
}

func ExampleSubscription_Delete() {
	ctx := context.Background()
	client, err := pubsub.NewClient(ctx, "project-id")
	if err != nil {
		// TODO: Handle error.
	}

	sub := client.Subscription("subName")
	if err := sub.Delete(ctx); err != nil {
		// TODO: Handle error.
	}
}

func ExampleSubscription_Exists() {
	ctx := context.Background()
	client, err := pubsub.NewClient(ctx, "project-id")
	if err != nil {
		// TODO: Handle error.
	}

	sub := client.Subscription("subName")
	ok, err := sub.Exists(ctx)
	if err != nil {
		// TODO: Handle error.
	}
	if !ok {
		// Subscription doesn't exist.
	}
}

func ExampleSubscription_Pull() {
	ctx := context.Background()
	client, err := pubsub.NewClient(ctx, "project-id")
	if err != nil {
		// TODO: Handle error.
	}

	sub := client.Subscription("subName")
	it, err := sub.Pull(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	// Ensure that the iterator is closed down cleanly.
	defer it.Stop()

	// Consume 10 messages.
	for i := 0; i < 10; i++ {
		m, err := it.Next()
		if err == pubsub.Done {
			// There are no more messages.  This will happen if it.Stop is called.
			break
		}
		if err != nil {
			// TODO: Handle error.
			break
		}
		fmt.Printf("message %d: %s\n", i, m.Data)

		// Acknowledge the message.
		m.Done(true)
	}
}
