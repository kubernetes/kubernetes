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
	"io/ioutil"
	"log"

	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/cloud"
	"google.golang.org/cloud/pubsub"
)

func Example_auth() context.Context {
	// Initialize an authorized context with Google Developers Console
	// JSON key. Read the google package examples to learn more about
	// different authorization flows you can use.
	// http://godoc.org/golang.org/x/oauth2/google
	jsonKey, err := ioutil.ReadFile("/path/to/json/keyfile.json")
	if err != nil {
		log.Fatal(err)
	}
	conf, err := google.JWTConfigFromJSON(
		jsonKey,
		pubsub.ScopeCloudPlatform,
		pubsub.ScopePubSub,
	)
	if err != nil {
		log.Fatal(err)
	}
	ctx := cloud.NewContext("project-id", conf.Client(oauth2.NoContext))
	// See the other samples to learn how to use the context.
	return ctx
}

func ExamplePublish() {
	ctx := Example_auth()

	msgIDs, err := pubsub.Publish(ctx, "topic1", &pubsub.Message{
		Data: []byte("hello world"),
	})
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Published a message with a message id: %s\n", msgIDs[0])
}

func ExamplePull() {
	ctx := Example_auth()

	// E.g. c.CreateSub("sub1", "topic1", time.Duration(0), "")
	msgs, err := pubsub.Pull(ctx, "sub1", 1)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("New message arrived: %v\n", msgs[0])
	if err := pubsub.Ack(ctx, "sub1", msgs[0].AckID); err != nil {
		log.Fatal(err)
	}
	log.Println("Acknowledged message")
}
