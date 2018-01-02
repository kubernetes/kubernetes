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

package pubsub

import (
	"fmt"
	"math/rand"
	"reflect"
	"sync"
	"testing"
	"time"

	"golang.org/x/net/context"

	"cloud.google.com/go/internal/testutil"
	"google.golang.org/api/option"
)

const timeout = time.Minute * 10
const ackDeadline = time.Second * 10

const batchSize = 100
const batches = 100

// messageCounter keeps track of how many times a given message has been received.
type messageCounter struct {
	mu     sync.Mutex
	counts map[string]int
	// A value is sent to recv each time Inc is called.
	recv chan struct{}
}

func (mc *messageCounter) Inc(msgID string) {
	mc.mu.Lock()
	mc.counts[msgID] += 1
	mc.mu.Unlock()
	mc.recv <- struct{}{}
}

// process pulls messages from an iterator and records them in mc.
func process(t *testing.T, it *Iterator, mc *messageCounter) {
	for {
		m, err := it.Next()
		if err == Done {
			return
		}

		if err != nil {
			t.Errorf("unexpected err from iterator: %v", err)
			return
		}
		mc.Inc(m.ID)
		// Simulate time taken to process m, while continuing to process more messages.
		go func() {
			// Some messages will need to have their ack deadline extended due to this delay.
			delay := rand.Intn(int(ackDeadline * 3))
			time.After(time.Duration(delay))
			m.Done(true)
		}()
	}
}

// newIter constructs a new Iterator.
func newIter(t *testing.T, ctx context.Context, sub *Subscription) *Iterator {
	it, err := sub.Pull(ctx)
	if err != nil {
		t.Fatalf("error constructing iterator: %v", err)
	}
	return it
}

// launchIter launches a number of goroutines to pull from the supplied Iterator.
func launchIter(t *testing.T, ctx context.Context, it *Iterator, mc *messageCounter, n int, wg *sync.WaitGroup) {
	for j := 0; j < n; j++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			process(t, it, mc)
		}()
	}
}

// iteratorLifetime controls how long iterators live for before they are stopped.
type iteratorLifetimes interface {
	// lifetimeChan should be called when an iterator is started. The
	// returned channel will send when the iterator should be stopped.
	lifetimeChan() <-chan time.Time
}

var immortal = &explicitLifetimes{}

// explicitLifetimes implements iteratorLifetime with hard-coded lifetimes, falling back
// to indefinite lifetimes when no explicit lifetimes remain.
type explicitLifetimes struct {
	mu        sync.Mutex
	lifetimes []time.Duration
}

func (el *explicitLifetimes) lifetimeChan() <-chan time.Time {
	el.mu.Lock()
	defer el.mu.Unlock()
	if len(el.lifetimes) == 0 {
		return nil
	}
	lifetime := el.lifetimes[0]
	el.lifetimes = el.lifetimes[1:]
	return time.After(lifetime)
}

// consumer consumes messages according to its configuration.
type consumer struct {
	// How many goroutines should pull from the subscription.
	iteratorsInFlight int
	// How many goroutines should pull from each iterator.
	concurrencyPerIterator int

	lifetimes iteratorLifetimes
}

// consume reads messages from a subscription, and keeps track of what it receives in mc.
// After consume returns, the caller should wait on wg to ensure that no more updates to mc will be made.
func (c *consumer) consume(t *testing.T, ctx context.Context, sub *Subscription, mc *messageCounter, wg *sync.WaitGroup, stop <-chan struct{}) {
	for i := 0; i < c.iteratorsInFlight; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				it := newIter(t, ctx, sub)
				launchIter(t, ctx, it, mc, c.concurrencyPerIterator, wg)

				select {
				case <-c.lifetimes.lifetimeChan():
					it.Stop()
				case <-stop:
					it.Stop()
					return
				}
			}

		}()
	}
}

// publish publishes many messages to topic, and returns the published message ids.
func publish(t *testing.T, ctx context.Context, topic *Topic) []string {
	var published []string
	msgs := make([]*Message, batchSize)
	for i := 0; i < batches; i++ {
		for j := 0; j < batchSize; j++ {
			text := fmt.Sprintf("msg %02d-%02d", i, j)
			msgs[j] = &Message{Data: []byte(text)}
		}
		ids, err := topic.Publish(ctx, msgs...)
		if err != nil {
			t.Errorf("Publish error: %v", err)
		}
		published = append(published, ids...)
	}
	return published
}

// diff returns counts of the differences between got and want.
func diff(got, want map[string]int) map[string]int {
	ids := make(map[string]struct{})
	for k := range got {
		ids[k] = struct{}{}
	}
	for k := range want {
		ids[k] = struct{}{}
	}

	gotWantCount := make(map[string]int)
	for k := range ids {
		if got[k] == want[k] {
			continue
		}
		desc := fmt.Sprintf("<got: %v ; want: %v>", got[k], want[k])
		gotWantCount[desc] += 1
	}
	return gotWantCount
}

// TestEndToEnd pumps many messages into a topic and tests that they are all delivered to each subscription for the topic.
// It also tests that messages are not unexpectedly redelivered.
func TestEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("Integration tests skipped in short mode")
	}
	ctx := context.Background()
	ts := testutil.TokenSource(ctx, ScopePubSub, ScopeCloudPlatform)
	if ts == nil {
		t.Skip("Integration tests skipped. See CONTRIBUTING.md for details")
	}

	now := time.Now()
	topicName := fmt.Sprintf("endtoend-%d", now.Unix())
	subPrefix := fmt.Sprintf("endtoend-%d", now.Unix())

	client, err := NewClient(ctx, testutil.ProjID(), option.WithTokenSource(ts))
	if err != nil {
		t.Fatalf("Creating client error: %v", err)
	}

	var topic *Topic
	if topic, err = client.CreateTopic(ctx, topicName); err != nil {
		t.Fatalf("CreateTopic error: %v", err)
	}
	defer topic.Delete(ctx)

	// Three subscriptions to the same topic.
	var subA, subB, subC *Subscription
	if subA, err = client.CreateSubscription(ctx, subPrefix+"-a", topic, ackDeadline, nil); err != nil {
		t.Fatalf("CreateSub error: %v", err)
	}
	defer subA.Delete(ctx)

	if subB, err = client.CreateSubscription(ctx, subPrefix+"-b", topic, ackDeadline, nil); err != nil {
		t.Fatalf("CreateSub error: %v", err)
	}
	defer subB.Delete(ctx)

	if subC, err = client.CreateSubscription(ctx, subPrefix+"-c", topic, ackDeadline, nil); err != nil {
		t.Fatalf("CreateSub error: %v", err)
	}
	defer subC.Delete(ctx)

	expectedCounts := make(map[string]int)
	for _, id := range publish(t, ctx, topic) {
		expectedCounts[id] = 1
	}

	// recv provides an indication that messages are still arriving.
	recv := make(chan struct{})

	// Keep track of the number of times each message (by message id) was
	// seen from each subscription.
	mcA := &messageCounter{counts: make(map[string]int), recv: recv}
	mcB := &messageCounter{counts: make(map[string]int), recv: recv}
	mcC := &messageCounter{counts: make(map[string]int), recv: recv}

	stopC := make(chan struct{})

	// We have three subscriptions to our topic.
	// Each subscription will get a copy of each pulished message.
	//
	// subA has just one iterator, while subB has two. The subB iterators
	// will each process roughly half of the messages for subB. All of
	// these iterators live until all messages have been consumed.  subC is
	// processed by a series of short-lived iterators.

	var wg sync.WaitGroup

	con := &consumer{
		concurrencyPerIterator: 1,
		iteratorsInFlight:      2,
		lifetimes:              immortal,
	}
	con.consume(t, ctx, subA, mcA, &wg, stopC)

	con = &consumer{
		concurrencyPerIterator: 1,
		iteratorsInFlight:      2,
		lifetimes:              immortal,
	}
	con.consume(t, ctx, subB, mcB, &wg, stopC)

	con = &consumer{
		concurrencyPerIterator: 1,
		iteratorsInFlight:      2,
		lifetimes: &explicitLifetimes{
			lifetimes: []time.Duration{ackDeadline, ackDeadline, ackDeadline / 2, ackDeadline / 2},
		},
	}
	con.consume(t, ctx, subC, mcC, &wg, stopC)

	go func() {
		timeoutC := time.After(timeout)
		// Every time this ticker ticks, we will check if we have received any
		// messages since the last time it ticked.  We check less frequently
		// than the ack deadline, so that we can detect if messages are
		// redelivered after having their ack deadline extended.
		checkQuiescence := time.NewTicker(ackDeadline * 3)
		defer checkQuiescence.Stop()

		var received bool
		for {
			select {
			case <-recv:
				received = true
			case <-checkQuiescence.C:
				if received {
					received = false
				} else {
					close(stopC)
					return
				}
			case <-timeoutC:
				t.Errorf("timed out")
				close(stopC)
				return
			}
		}
	}()
	wg.Wait()

	for _, mc := range []*messageCounter{mcA, mcB, mcC} {
		if got, want := mc.counts, expectedCounts; !reflect.DeepEqual(got, want) {
			t.Errorf("message counts: %v\n", diff(got, want))
		}
	}
}
