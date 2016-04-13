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

// Package main contains a client which pulls messages from a subscription and prints them.
package main

import (
	"flag"
	"fmt"
	"log"
	"time"

	"golang.org/x/net/context"

	"google.golang.org/cloud/pubsub"
)

var (
	projID     = flag.String("p", "", "The ID of your Google Cloud project.")
	subName    = flag.String("s", "", "The name of the subscription to pull from")
	numConsume = flag.Int("n", 10, "The number of messages to consume")
)

func main() {
	flag.Parse()

	if *projID == "" {
		log.Fatal("-p is required")
	}
	if *subName == "" {
		log.Fatal("-s is required")
	}

	ctx := context.Background()

	client, err := pubsub.NewClient(ctx, *projID)
	if err != nil {
		log.Fatalf("creating pubsub client: %v", err)
	}

	sub := client.Subscription(*subName)

	it, err := sub.Pull(ctx, pubsub.MaxExtension(time.Minute))
	if err != nil {
		fmt.Printf("error constructing iterator: %v", err)
		return
	}
	defer it.Stop()

	for i := 0; i < *numConsume; i++ {
		m, err := it.Next()
		if err != nil {
			fmt.Printf("advancing iterator: %v", err)
			break
		}
		fmt.Printf("got message: %v\n", string(m.Data))
		m.Done(true)
	}
}
