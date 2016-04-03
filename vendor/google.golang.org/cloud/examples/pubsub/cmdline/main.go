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

// Package main contains a simple command line tool for Cloud Pub/Sub
// Cloud Pub/Sub docs: https://cloud.google.com/pubsub/docs
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"time"

	"golang.org/x/net/context"

	"google.golang.org/cloud/pubsub"
)

var (
	projID    = flag.String("p", "", "The ID of your Google Cloud project.")
	reportMPS = flag.Bool("report", false, "Reports the incoming/outgoing message rate in msg/sec if set.")
	size      = flag.Int("size", 10, "Batch size for publish_messages subcommand.")
)

const (
	usage = `Available arguments are:
    create_topic <name>
    topic_exists <name>
    delete_topic <name>
    list_topic_subscriptions <name>
    list_topics
    create_subscription <name> <linked_topic>
    show_subscription <name>
    subscription_exists <name>
    delete_subscription <name>
    list_subscriptions
    publish <topic> <message>
    pull_messages <subscription> <numworkers>
    publish_messages <topic> <numworkers>
`
	tick = 1 * time.Second
)

func usageAndExit(msg string) {
	fmt.Fprintln(os.Stderr, msg)
	fmt.Println("Flags:")
	flag.PrintDefaults()
	fmt.Fprint(os.Stderr, usage)
	os.Exit(2)
}

// Check the length of the arguments.
func checkArgs(argv []string, min int) {
	if len(argv) < min {
		usageAndExit("Missing arguments")
	}
}

func createTopic(client *pubsub.Client, argv []string) {
	checkArgs(argv, 2)
	topic := argv[1]
	_, err := client.NewTopic(context.Background(), topic)
	if err != nil {
		log.Fatalf("Creating topic failed: %v", err)
	}
	fmt.Printf("Topic %s was created.\n", topic)
}

func listTopics(client *pubsub.Client, argv []string) {
	checkArgs(argv, 1)
	topics, err := client.Topics(context.Background())
	if err != nil {
		log.Fatalf("Listing topics failed: %v", err)
	}
	for _, t := range topics {
		fmt.Println(t.Name())
	}
}

func listTopicSubscriptions(client *pubsub.Client, argv []string) {
	checkArgs(argv, 2)
	topic := argv[1]
	subs, err := client.Topic(topic).Subscriptions(context.Background())
	if err != nil {
		log.Fatalf("Listing subscriptions failed: %v", err)
	}
	for _, s := range subs {
		fmt.Println(s.Name())
	}
}

func checkTopicExists(client *pubsub.Client, argv []string) {
	checkArgs(argv, 1)
	topic := argv[1]
	exists, err := client.Topic(topic).Exists(context.Background())
	if err != nil {
		log.Fatalf("Checking topic exists failed: %v", err)
	}
	fmt.Println(exists)
}

func deleteTopic(client *pubsub.Client, argv []string) {
	checkArgs(argv, 2)
	topic := argv[1]
	err := client.Topic(topic).Delete(context.Background())
	if err != nil {
		log.Fatalf("Deleting topic failed: %v", err)
	}
	fmt.Printf("Topic %s was deleted.\n", topic)
}

func createSubscription(client *pubsub.Client, argv []string) {
	checkArgs(argv, 3)
	sub := argv[1]
	topic := argv[2]
	_, err := client.Topic(topic).Subscribe(context.Background(), sub, 0, nil)
	if err != nil {
		log.Fatalf("Creating Subscription failed: %v", err)
	}
	fmt.Printf("Subscription %s was created.\n", sub)
}

func showSubscription(client *pubsub.Client, argv []string) {
	checkArgs(argv, 2)
	sub := argv[1]
	conf, err := client.Subscription(sub).Config(context.Background())
	if err != nil {
		log.Fatalf("Getting Subscription failed: %v", err)
	}
	fmt.Printf("%+v\n", conf)
	exists, err := conf.Topic.Exists(context.Background())
	if err != nil {
		log.Fatalf("Checking whether topic exists: %v", err)
	}
	if !exists {
		fmt.Println("The topic for this subscription has been deleted.\n")
	}
}

func checkSubscriptionExists(client *pubsub.Client, argv []string) {
	checkArgs(argv, 1)
	sub := argv[1]
	exists, err := client.Subscription(sub).Exists(context.Background())
	if err != nil {
		log.Fatalf("Checking subscription exists failed: %v", err)
	}
	fmt.Println(exists)
}

func deleteSubscription(client *pubsub.Client, argv []string) {
	checkArgs(argv, 2)
	sub := argv[1]
	err := client.Subscription(sub).Delete(context.Background())
	if err != nil {
		log.Fatalf("Deleting Subscription failed: %v", err)
	}
	fmt.Printf("Subscription %s was deleted.\n", sub)
}

func listSubscriptions(client *pubsub.Client, argv []string) {
	checkArgs(argv, 1)
	subs, err := client.Subscriptions(context.Background())
	if err != nil {
		log.Fatalf("Listing subscriptions failed: %v", err)
	}
	for _, s := range subs {
		fmt.Println(s.Name())
	}
}

func publish(client *pubsub.Client, argv []string) {
	checkArgs(argv, 3)
	topic := argv[1]
	message := argv[2]
	msgIDs, err := client.Topic(topic).Publish(context.Background(), &pubsub.Message{
		Data: []byte(message),
	})
	if err != nil {
		log.Fatalf("Publish failed, %v", err)
	}
	fmt.Printf("Message '%s' published to topic %s and the message id is %s\n", message, topic, msgIDs[0])
}

// reporter maintains a counter and reports stats about the rate of increase of that counter.
type reporter struct {
	reportTitle string
	lastC       uint64
	c           uint64
	count       chan int

	// Close done to shut down reporter.
	done chan struct{}
}

// newReporter constructs a reporter which logs stats if --report is true.
// Users must call Stop once the reporter is no longer needed.
func newReporter(reportTitle string) *reporter {
	rep := &reporter{reportTitle: reportTitle}
	rep.start()
	return rep
}

func (r *reporter) start() {
	ticker := time.NewTicker(tick)
	r.done = make(chan struct{})
	r.count = make(chan int, 1024)
	go func() {
		defer func() {
			ticker.Stop()
		}()
		for {
			select {
			case <-ticker.C:
				n := r.c - r.lastC
				r.lastC = r.c
				mps := n / uint64(tick/time.Second)
				if *reportMPS {
					log.Printf("%s ~%d msgs/s, total: %d", r.reportTitle, mps, r.c)
				}
			case n := <-r.count:
				r.c += uint64(n)
			case <-r.done:
				return
			}
		}
	}()
}

// Inc increments the message count by n.
func (r *reporter) Inc(n int) {
	r.count <- n
}

// Stop Stops the reporting that was started by Start.
func (r *reporter) Stop() {
	close(r.done)
}

var quit chan os.Signal

func init() {
	quit = make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt)
}

func shouldQuit() bool {
	select {
	case <-quit:
		signal.Stop(quit)
		close(quit)
		return true
	default:
		return false
	}
}

// genMessages generates a batch of messages to send.
func genMessages(prefix string) []*pubsub.Message {
	msgs := make([]*pubsub.Message, *size)
	for i := 0; i < *size; i++ {
		msgs[i] = &pubsub.Message{
			Data: []byte(fmt.Sprintf("%s Message: %d", prefix, i)),
		}
	}
	return msgs
}

// publish publishes a series of messages to the named topic.
func publishMessageBatches(client *pubsub.Client, topicName string, workerid int, rep *reporter) {
	var r uint64
	topic := client.Topic(topicName)
	for !shouldQuit() {
		msgPrefix := fmt.Sprintf("Worker: %d, Round: %d,", workerid, r)
		if _, err := topic.Publish(context.Background(), genMessages(msgPrefix)...); err != nil {
			log.Printf("Publish failed, %v\n", err)
			return
		}
		r++
		rep.Inc(*size)
	}
}

func publishMessages(client *pubsub.Client, argv []string) {
	checkArgs(argv, 3)
	topic := argv[1]
	workers, err := strconv.Atoi(argv[2])
	if err != nil {
		log.Fatalf("Atoi failed, %v", err)
	}
	rep := newReporter("Sent")
	defer rep.Stop()

	var wg sync.WaitGroup
	for i := 0; i < int(workers); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			publishMessageBatches(client, topic, i, rep)
		}()
	}
	wg.Wait()
}

// processMessages reads Messages from msgs and processes them, until mgss is closed.
// It calls Done on each Message that is read from msgs.
func processMessages(msgs <-chan *pubsub.Message, rep *reporter, printMsg bool) {
	for m := range msgs {
		if printMsg {
			fmt.Printf("Got a message: %s\n", m.Data)
		}
		rep.Inc(1)
		m.Done(true)
	}
}

// pullMessages reads messages from a subscription, and farms them out to a
// number of goroutines for processing.
func pullMessages(client *pubsub.Client, argv []string) {
	checkArgs(argv, 3)
	sub := client.Subscription(argv[1])

	workers, err := strconv.Atoi(argv[2])
	if err != nil {
		log.Fatalf("Atoi failed, %v", err)
	}

	rep := newReporter("Received")
	defer rep.Stop()

	msgs := make(chan *pubsub.Message)
	for i := 0; i < int(workers); i++ {
		go processMessages(msgs, rep, !*reportMPS)
	}

	it, err := sub.Pull(context.Background(), pubsub.MaxExtension(time.Minute))
	if err != nil {
		log.Fatalf("failed to construct iterator: %v", err)
	}
	defer it.Stop()

	for !shouldQuit() {
		m, err := it.Next()
		if err != nil {
			log.Fatalf("error reading from iterator: %v", err)
		}
		msgs <- m
	}

	// Shut down all processMessages goroutines.
	close(msgs)

	// The deferred call to it.Stop will block until each m.Done has been
	// called on each message.
}

// This example demonstrates calling the Cloud Pub/Sub API.
//
// Before running this example, be sure to enable Cloud Pub/Sub
// service on your project in Developer Console at:
// https://console.developers.google.com/
//
// Unless you run this sample on Compute Engine instance, please
// create a new service account and download a JSON key file for it at
// the developer console: https://console.developers.google.com/
//
// It has the following subcommands:
//
//  create_topic <name>
//  delete_topic <name>
//  create_subscription <name> <linked_topic>
//  delete_subscription <name>
//  publish <topic> <message>
//  pull_messages <subscription> <numworkers>
//  publish_messages <topic> <numworkers>
//
// You can choose any names for topic and subscription as long as they
// follow the naming rule described at:
// https://cloud.google.com/pubsub/overview#names
//
// You can create/delete topics/subscriptions by self-explanatory
// subcommands.
//
// The "publish" subcommand is for publishing a single message to a
// specified Cloud Pub/Sub topic.
//
// The "pull_messages" subcommand is for continuously pulling messages
// from a specified Cloud Pub/Sub subscription with specified number
// of workers.
//
// The "publish_messages" subcommand is for continuously publishing
// messages to a specified Cloud Pub/Sub topic with specified number
// of workers.
func main() {
	flag.Parse()
	argv := flag.Args()
	checkArgs(argv, 1)
	if *projID == "" {
		usageAndExit("Please specify Project ID.")
	}
	client, err := pubsub.NewClient(context.Background(), *projID)
	if err != nil {
		log.Fatalf("creating pubsub client: %v", err)
	}

	commands := map[string]func(client *pubsub.Client, argv []string){
		"create_topic":             createTopic,
		"delete_topic":             deleteTopic,
		"list_topics":              listTopics,
		"list_topic_subscriptions": listTopicSubscriptions,
		"topic_exists":             checkTopicExists,
		"create_subscription":      createSubscription,
		"show_subscription":        showSubscription,
		"delete_subscription":      deleteSubscription,
		"subscription_exists":      checkSubscriptionExists,
		"list_subscriptions":       listSubscriptions,
		"publish":                  publish,
		"publish_messages":         publishMessages,
		"pull_messages":            pullMessages,
	}
	subcommand := argv[0]
	if f, ok := commands[subcommand]; ok {
		f(client, argv)
	} else {
		usageAndExit(fmt.Sprintf("Function not found for %s", subcommand))
	}
}
